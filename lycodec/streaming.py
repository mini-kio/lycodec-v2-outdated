import torch
import torch.nn.functional as F
import numpy as np
from typing import Iterator, Optional, Tuple
from contextlib import contextmanager
import threading
import queue
import time

from .inference import LyCodec
from .audio import HOP_LENGTH, SAMPLE_RATE

class StreamingDecoder:
    """
    Real-time streaming decoder for LyCodec
    Leverages center=False STFT structure for low-latency streaming
    """
    
    def __init__(self, 
                 model_path: str,
                 device: Optional[str] = None,
                 chunk_size: int = 4410,  # 0.1 second at 44.1kHz
                 overlap_ratio: float = 0.5,
                 buffer_size: int = 8):
        
        # IMPROVED: Validate chunk and overlap parameters
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError(f"overlap_ratio must be in [0, 1), got {overlap_ratio}")
        
        overlap_length = int(chunk_size * overlap_ratio)
        if overlap_length >= chunk_size:
            raise ValueError(f"Overlap length ({overlap_length}) must be < chunk_size ({chunk_size})")
        
        # Initialize base codec
        self.codec = LyCodec(
            model_path=model_path,
            device=device,
            half_precision=True,
            max_chunk_length=chunk_size * 10,  # Process 1 second chunks internally
            overlap_length=overlap_length
        )
        
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.overlap_length = overlap_length  # Store for safety checks
        self.buffer_size = buffer_size
        self.sample_rate = SAMPLE_RATE
        
        # Streaming state
        self.is_streaming = False
        self.input_queue = queue.Queue(maxsize=buffer_size)
        self.output_queue = queue.Queue(maxsize=buffer_size)
        self.worker_thread = None
        
        # Overlap-add state for smooth streaming - IMPROVED size safety
        self.overlap_buffer = np.zeros((2, overlap_length), dtype=np.float32)
        
        print(f"StreamingDecoder initialized: {chunk_size/SAMPLE_RATE*1000:.1f}ms latency, "
              f"{overlap_length/SAMPLE_RATE*1000:.1f}ms overlap")
    
    def _worker_loop(self):
        """Background worker for continuous decoding"""
        while self.is_streaming:
            try:
                # Get latent chunk from input queue (blocking with timeout)
                latent_chunk = self.input_queue.get(timeout=0.1)
                
                if latent_chunk is None:  # Poison pill to stop
                    break
                
                # Decode chunk
                start_time = time.time()
                audio_chunk = self.codec.decode(latent_chunk)  # [2, T]
                decode_time = time.time() - start_time
                
                # IMPROVED: Apply overlap-add with safety checks
                if isinstance(audio_chunk, torch.Tensor):
                    audio_chunk = audio_chunk.cpu().numpy()
                
                # Ensure audio_chunk has correct shape
                if audio_chunk.ndim == 1:
                    audio_chunk = audio_chunk.reshape(1, -1)
                if audio_chunk.shape[0] == 1:
                    audio_chunk = np.repeat(audio_chunk, 2, axis=0)  # Convert mono to stereo
                elif audio_chunk.shape[0] > 2:
                    audio_chunk = audio_chunk[:2]  # Keep only first 2 channels
                
                overlap_len = self.overlap_buffer.shape[-1]
                
                # Safety check: ensure audio chunk is long enough for overlap
                if audio_chunk.shape[-1] >= overlap_len:
                    # Add overlap from previous chunk
                    if overlap_len > 0:
                        audio_chunk[:, :overlap_len] += self.overlap_buffer
                    
                    # Store overlap for next chunk with fade
                    if audio_chunk.shape[-1] >= overlap_len and overlap_len > 0:
                        self.overlap_buffer = audio_chunk[:, -overlap_len:].copy()
                        # Fade out overlap region for smoother transitions
                        fade = np.linspace(1.0, 0.0, overlap_len)
                        self.overlap_buffer *= fade
                else:
                    # Handle edge case: chunk too short for overlap
                    print(f"Warning: Audio chunk ({audio_chunk.shape[-1]}) shorter than overlap ({overlap_len})")
                    # Skip overlap-add for this chunk
                    pass
                
                # Add timing info for monitoring
                chunk_info = {
                    'audio': audio_chunk,
                    'decode_time_ms': decode_time * 1000,
                    'timestamp': time.time()
                }
                
                # Put result in output queue (non-blocking)
                try:
                    self.output_queue.put(chunk_info, timeout=0.01)
                except queue.Full:
                    # Drop frame if output queue is full (real-time constraint)
                    print("Warning: Dropped audio chunk due to full output queue")
                
                self.input_queue.task_done()
                
            except queue.Empty:
                continue  # No input available, continue loop
            except Exception as e:
                print(f"Error in streaming worker: {e}")
                break
    
    def start_streaming(self):
        """Start the streaming decoder"""
        if self.is_streaming:
            return
        
        self.is_streaming = True
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        print("Streaming decoder started")
    
    def stop_streaming(self):
        """Stop the streaming decoder"""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        
        # Send poison pill to stop worker
        try:
            self.input_queue.put(None, timeout=0.1)
        except queue.Full:
            pass
        
        # Wait for worker to finish
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
        
        # Clear queues
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break
        
        print("Streaming decoder stopped")
    
    def put_latent_chunk(self, latent_chunk: torch.Tensor) -> bool:
        """
        Put a latent chunk for decoding (non-blocking)
        Returns True if successful, False if queue is full
        """
        try:
            self.input_queue.put(latent_chunk, timeout=0.001)
            return True
        except queue.Full:
            return False
    
    def get_audio_chunk(self, timeout: float = 0.1) -> Optional[dict]:
        """
        Get decoded audio chunk (blocking with timeout)
        Returns dict with 'audio', 'decode_time_ms', 'timestamp' or None if timeout
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stream_decode_generator(self, latent_stream: Iterator[torch.Tensor]) -> Iterator[np.ndarray]:
        """
        Generator-based streaming interface
        Yields audio chunks as they become available
        """
        self.start_streaming()
        
        try:
            # Feed latent chunks to decoder
            for latent_chunk in latent_stream:
                if not self.put_latent_chunk(latent_chunk):
                    print("Warning: Input queue full, skipping chunk")
                
                # Yield available audio chunks
                while True:
                    chunk_info = self.get_audio_chunk(timeout=0.001)
                    if chunk_info is None:
                        break
                    yield chunk_info['audio']
        
        finally:
            self.stop_streaming()
    
    @contextmanager
    def streaming_session(self):
        """Context manager for streaming sessions"""
        self.start_streaming()
        try:
            yield self
        finally:
            self.stop_streaming()
    
    def get_latency_ms(self) -> float:
        """Get theoretical minimum latency in milliseconds"""
        return (self.chunk_size / self.sample_rate) * 1000
    
    def get_stats(self) -> dict:
        """Get streaming statistics"""
        return {
            'chunk_size_ms': self.get_latency_ms(),
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'is_streaming': self.is_streaming,
            'overlap_ratio': self.overlap_ratio
        }

# Convenience function for quick streaming setup
def create_streaming_decoder(model_path: str, latency_ms: float = 100) -> StreamingDecoder:
    """
    Create a streaming decoder with specified target latency
    
    Args:
        model_path: Path to trained model
        latency_ms: Target latency in milliseconds (default: 100ms)
    
    Returns:
        StreamingDecoder instance
    """
    # IMPROVED: Validate latency parameter
    if latency_ms <= 0:
        raise ValueError(f"latency_ms must be positive, got {latency_ms}")
    
    chunk_size = int(SAMPLE_RATE * latency_ms / 1000)
    
    # IMPROVED: Ensure minimum chunk size for stability
    min_chunk_size = 1024  # ~23ms at 44.1kHz
    if chunk_size < min_chunk_size:
        print(f"Warning: Requested latency ({latency_ms}ms) too low, using minimum ({min_chunk_size/SAMPLE_RATE*1000:.1f}ms)")
        chunk_size = min_chunk_size
    
    return StreamingDecoder(
        model_path=model_path,
        chunk_size=chunk_size,
        overlap_ratio=0.5
    )
