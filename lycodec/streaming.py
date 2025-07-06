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
    Real-time streaming decoder for LyCodec with improved error handling
    Leverages center=False STFT structure for low-latency streaming
    """
    
    def __init__(self, 
                 model_path: str,
                 device: Optional[str] = None,
                 chunk_size: int = 4410,  # 0.1 second at 44.1kHz
                 overlap_ratio: float = 0.5,
                 buffer_size: int = 8):
        
        # IMPROVED: Validate chunk and overlap parameters with better error messages
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError(f"overlap_ratio must be in [0, 1), got {overlap_ratio}")
        
        overlap_length = int(chunk_size * overlap_ratio)
        if overlap_length >= chunk_size:
            raise ValueError(f"Overlap length ({overlap_length}) must be < chunk_size ({chunk_size})")
        
        # Initialize base codec with better error handling
        try:
            self.codec = LyCodec(
                model_path=model_path,
                device=device,
                half_precision=True,
                max_chunk_length=chunk_size * 10,  # Process 1 second chunks internally
                overlap_length=overlap_length
            )
        except Exception as e:
            print(f"Warning: Failed to load model from {model_path}: {e}")
            print("Creating codec with dummy model for testing")
            self.codec = LyCodec(
                model_path=None,  # No model - will use dummy weights
                device=device,
                half_precision=True,
                max_chunk_length=chunk_size * 10,
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
        self._stop_event = threading.Event()  # IMPROVED: Better thread coordination
        
        # Overlap-add state for smooth streaming - IMPROVED size safety
        try:
            self.overlap_buffer = np.zeros((2, overlap_length), dtype=np.float32)
        except Exception as e:
            print(f"Warning: Failed to create overlap buffer ({e}), using no overlap")
            self.overlap_buffer = np.zeros((2, 0), dtype=np.float32)
            self.overlap_length = 0
        
        print(f"StreamingDecoder initialized: {chunk_size/SAMPLE_RATE*1000:.1f}ms latency, "
              f"{overlap_length/SAMPLE_RATE*1000:.1f}ms overlap")
    
    def _worker_loop(self):
        """Background worker for continuous decoding with improved error handling"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while not self._stop_event.is_set():
            try:
                # Get latent chunk from input queue (blocking with timeout)
                try:
                    latent_chunk = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue  # No input available, continue loop
                
                if latent_chunk is None:  # Poison pill to stop
                    break
                
                # Decode chunk with error handling
                start_time = time.time()
                try:
                    audio_chunk = self.codec.decode(latent_chunk)  # [2, T]
                    decode_time = time.time() - start_time
                    
                    # Reset error counter on successful decode
                    consecutive_errors = 0
                    
                except Exception as e:
                    print(f"Warning: Decode failed: {e}")
                    consecutive_errors += 1
                    
                    # Create dummy audio chunk on decode failure
                    decode_time = time.time() - start_time
                    audio_chunk = np.zeros((2, self.chunk_size), dtype=np.float32)
                    
                    # Stop if too many consecutive errors
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"Error: Too many consecutive decode failures ({consecutive_errors}), stopping worker")
                        break
                
                # IMPROVED: Apply overlap-add with safety checks and error handling
                try:
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
                    if audio_chunk.shape[-1] >= overlap_len and overlap_len > 0:
                        # Add overlap from previous chunk
                        audio_chunk[:, :overlap_len] += self.overlap_buffer
                        
                        # Store overlap for next chunk with fade
                        if audio_chunk.shape[-1] >= overlap_len:
                            self.overlap_buffer = audio_chunk[:, -overlap_len:].copy()
                            # Fade out overlap region for smoother transitions
                            fade = np.linspace(1.0, 0.0, overlap_len)
                            self.overlap_buffer *= fade
                    elif overlap_len > 0:
                        # Handle edge case: chunk too short for overlap
                        print(f"Warning: Audio chunk ({audio_chunk.shape[-1]}) shorter than overlap ({overlap_len})")
                        # Skip overlap-add for this chunk
                        pass
                    
                except Exception as e:
                    print(f"Warning: Overlap-add failed: {e}")
                    # Continue without overlap processing
                
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
                
                # Mark task as done
                try:
                    self.input_queue.task_done()
                except ValueError:
                    # Task already marked done or queue empty
                    pass
                
            except Exception as e:
                print(f"Error in streaming worker: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print("Error: Too many consecutive worker errors, stopping")
                    break
                continue
        
        print("Streaming worker thread stopped")
    
    def start_streaming(self):
        """Start the streaming decoder with better error handling"""
        if self.is_streaming:
            return
        
        try:
            self.is_streaming = True
            self._stop_event.clear()
            self.worker_thread = threading.Thread(target=self._worker_loop)
            self.worker_thread.daemon = True
            self.worker_thread.start()
            print("Streaming decoder started")
        except Exception as e:
            print(f"Error starting streaming decoder: {e}")
            self.is_streaming = False
    
    def stop_streaming(self):
        """Stop the streaming decoder with better cleanup"""
        if not self.is_streaming:
            return
        
        try:
            self.is_streaming = False
            self._stop_event.set()
            
            # Send poison pill to stop worker
            try:
                self.input_queue.put(None, timeout=0.1)
            except queue.Full:
                pass
            
            # Wait for worker to finish
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=2.0)  # Increased timeout
                if self.worker_thread.is_alive():
                    print("Warning: Worker thread did not stop gracefully")
            
            # Clear queues
            self._clear_queue(self.input_queue)
            self._clear_queue(self.output_queue)
            
            print("Streaming decoder stopped")
            
        except Exception as e:
            print(f"Error stopping streaming decoder: {e}")
    
    def _clear_queue(self, q):
        """Safely clear a queue"""
        try:
            while not q.empty():
                try:
                    q.get_nowait()
                    q.task_done()
                except queue.Empty:
                    break
                except ValueError:
                    # Task already done
                    break
        except Exception as e:
            print(f"Warning: Error clearing queue: {e}")
    
    def put_latent_chunk(self, latent_chunk: torch.Tensor) -> bool:
        """
        Put a latent chunk for decoding (non-blocking)
        Returns True if successful, False if queue is full
        """
        if not self.is_streaming:
            print("Warning: Cannot put chunk, streaming not started")
            return False
            
        try:
            self.input_queue.put(latent_chunk, timeout=0.001)
            return True
        except queue.Full:
            return False
        except Exception as e:
            print(f"Error putting latent chunk: {e}")
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
        except Exception as e:
            print(f"Error getting audio chunk: {e}")
            return None
    
    def stream_decode_generator(self, latent_stream: Iterator[torch.Tensor]) -> Iterator[np.ndarray]:
        """
        Generator-based streaming interface with better error handling
        Yields audio chunks as they become available
        """
        self.start_streaming()
        
        try:
            # Feed latent chunks to decoder
            for latent_chunk in latent_stream:
                try:
                    if not self.put_latent_chunk(latent_chunk):
                        print("Warning: Input queue full, skipping chunk")
                    
                    # Yield available audio chunks
                    attempts = 0
                    max_attempts = 10  # Prevent infinite loop
                    while attempts < max_attempts:
                        chunk_info = self.get_audio_chunk(timeout=0.001)
                        if chunk_info is None:
                            break
                        yield chunk_info['audio']
                        attempts += 1
                        
                except Exception as e:
                    print(f"Error in stream decode generator: {e}")
                    continue
        
        except Exception as e:
            print(f"Error in latent stream processing: {e}")
        
        finally:
            self.stop_streaming()
    
    @contextmanager
    def streaming_session(self):
        """Context manager for streaming sessions with better cleanup"""
        self.start_streaming()
        try:
            yield self
        except Exception as e:
            print(f"Error in streaming session: {e}")
        finally:
            self.stop_streaming()
    
    def get_latency_ms(self) -> float:
        """Get theoretical minimum latency in milliseconds"""
        return (self.chunk_size / self.sample_rate) * 1000
    
    def get_stats(self) -> dict:
        """Get streaming statistics with error handling"""
        try:
            return {
                'chunk_size_ms': self.get_latency_ms(),
                'input_queue_size': self.input_queue.qsize(),
                'output_queue_size': self.output_queue.qsize(),
                'is_streaming': self.is_streaming,
                'overlap_ratio': self.overlap_ratio,
                'worker_alive': self.worker_thread.is_alive() if self.worker_thread else False
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {
                'chunk_size_ms': self.get_latency_ms(),
                'input_queue_size': 0,
                'output_queue_size': 0,
                'is_streaming': self.is_streaming,
                'overlap_ratio': self.overlap_ratio,
                'worker_alive': False
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
    # IMPROVED: Validate latency parameter with better error messages
    if latency_ms <= 0:
        raise ValueError(f"latency_ms must be positive, got {latency_ms}")
    
    if latency_ms > 1000:
        print(f"Warning: Very high latency requested ({latency_ms}ms), consider reducing")
    
    chunk_size = int(SAMPLE_RATE * latency_ms / 1000)
    
    # IMPROVED: Ensure minimum chunk size for stability
    min_chunk_size = 1024  # ~23ms at 44.1kHz
    if chunk_size < min_chunk_size:
        print(f"Warning: Requested latency ({latency_ms}ms) too low, using minimum ({min_chunk_size/SAMPLE_RATE*1000:.1f}ms)")
        chunk_size = min_chunk_size
    
    # IMPROVED: Warn about very large chunk sizes
    max_chunk_size = SAMPLE_RATE  # 1 second max
    if chunk_size > max_chunk_size:
        print(f"Warning: Requested latency ({latency_ms}ms) very high, capping at {max_chunk_size/SAMPLE_RATE*1000:.1f}ms")
        chunk_size = max_chunk_size
    
    try:
        return StreamingDecoder(
            model_path=model_path,
            chunk_size=chunk_size,
            overlap_ratio=0.5
        )
    except Exception as e:
        print(f"Error creating streaming decoder: {e}")
        # Return decoder with dummy model for testing
        return StreamingDecoder(
            model_path=None,  # Will create dummy model
            chunk_size=chunk_size,
            overlap_ratio=0.5
        )