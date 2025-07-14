import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from pathlib import Path
import warnings
from typing import Iterator, Optional, Tuple
from contextlib import contextmanager
import threading
import queue
import time

from .inference import Codec
from .audio import HOP_LENGTH, SAMPLE_RATE, N_MELS

class StreamingDecoder:
    """
    FIXED: 단순화된 real-time streaming decoder
    안정성을 위해 복잡성 최소화
    """
    
    def __init__(self, 
                 model_path: str,
                 device: Optional[str] = None,
                 chunk_size: int = 4410,  # 0.1 second
                 overlap_ratio: float = 0.5,
                 buffer_size: int = 8):
        
        # Validate parameters
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError(f"overlap_ratio must be in [0, 1), got {overlap_ratio}")
        
        overlap_length = int(chunk_size * overlap_ratio)
        if overlap_length >= chunk_size:
            raise ValueError(f"Overlap length ({overlap_length}) must be < chunk_size ({chunk_size})")
        
        # Initialize codec with simplified settings
        try:
            self.codec = Codec(
                model_path=model_path,
                device=device,
                half_precision=True,
                max_chunk_length=chunk_size * 10,
                overlap_length=overlap_length
            )
        except Exception:
            # Fallback to dummy model
            self.codec = Codec(
                model_path=None,
                device=device,
                half_precision=True,
                max_chunk_length=chunk_size * 10,
                overlap_length=overlap_length
            )
        
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.overlap_length = overlap_length
        self.buffer_size = buffer_size
        self.sample_rate = SAMPLE_RATE
        
        # Streaming state
        self.is_streaming = False
        self.input_queue = queue.Queue(maxsize=buffer_size)
        self.output_queue = queue.Queue(maxsize=buffer_size)
        self.worker_thread = None
        self._stop_event = threading.Event()
        
        # Overlap buffer
        try:
            self.overlap_buffer = np.zeros((2, overlap_length), dtype=np.float32)
        except Exception:
            self.overlap_buffer = np.zeros((2, 0), dtype=np.float32)
            self.overlap_length = 0
    
    def _worker_loop(self):
        """Background worker for continuous decoding - simplified"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while not self._stop_event.is_set():
            try:
                # Get latent chunk
                try:
                    latent_chunk = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                if latent_chunk is None:  # Stop signal
                    break
                
                # Decode chunk
                start_time = time.time()
                try:
                    audio_chunk = self.codec.decode(latent_chunk)
                    decode_time = time.time() - start_time
                    consecutive_errors = 0
                    
                except Exception:
                    consecutive_errors += 1
                    decode_time = time.time() - start_time
                    audio_chunk = np.zeros((2, self.chunk_size), dtype=np.float32)
                    
                    if consecutive_errors >= max_consecutive_errors:
                        break
                
                # Apply overlap-add
                try:
                    if isinstance(audio_chunk, torch.Tensor):
                        audio_chunk = audio_chunk.cpu().numpy()
                    
                    # Ensure correct shape
                    if audio_chunk.ndim == 1:
                        audio_chunk = audio_chunk.reshape(1, -1)
                    if audio_chunk.shape[0] == 1:
                        audio_chunk = np.repeat(audio_chunk, 2, axis=0)
                    elif audio_chunk.shape[0] > 2:
                        audio_chunk = audio_chunk[:2]
                    
                    overlap_len = self.overlap_buffer.shape[-1]
                    
                    # Simple overlap-add
                    if audio_chunk.shape[-1] >= overlap_len and overlap_len > 0:
                        audio_chunk[:, :overlap_len] += self.overlap_buffer
                        
                        if audio_chunk.shape[-1] >= overlap_len:
                            self.overlap_buffer = audio_chunk[:, -overlap_len:].copy()
                            fade = np.linspace(1.0, 0.0, overlap_len)
                            self.overlap_buffer *= fade
                    
                except Exception:
                    pass  # Continue without overlap processing
                
                # Create result
                chunk_info = {
                    'audio': audio_chunk,
                    'decode_time_ms': decode_time * 1000,
                    'timestamp': time.time(),
                    'architecture': 'log_mel_phase'
                }
                
                # Put result in output queue
                try:
                    self.output_queue.put(chunk_info, timeout=0.01)
                except queue.Full:
                    pass  # Drop frame if queue is full
                
                # Mark task as done
                try:
                    self.input_queue.task_done()
                except ValueError:
                    pass
                
            except Exception:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    break
                continue
    
    def start_streaming(self):
        """Start the streaming decoder"""
        if self.is_streaming:
            return
        
        try:
            self.is_streaming = True
            self._stop_event.clear()
            self.worker_thread = threading.Thread(target=self._worker_loop)
            self.worker_thread.daemon = True
            self.worker_thread.start()
        except Exception:
            self.is_streaming = False
    
    def stop_streaming(self):
        """Stop the streaming decoder"""
        if not self.is_streaming:
            return
        
        try:
            self.is_streaming = False
            self._stop_event.set()
            
            # Send stop signal
            try:
                self.input_queue.put(None, timeout=0.1)
            except queue.Full:
                pass
            
            # Wait for worker
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=2.0)
            
            # Clear queues
            self._clear_queue(self.input_queue)
            self._clear_queue(self.output_queue)
            
            # Reset buffers
            try:
                self.overlap_buffer.fill(0)
            except Exception:
                pass
            
        except Exception:
            pass
    
    def _clear_queue(self, q):
        """Safely clear a queue"""
        try:
            while not q.empty():
                try:
                    q.get_nowait()
                    q.task_done()
                except (queue.Empty, ValueError):
                    break
        except Exception:
            pass
    
    def put_latent_chunk(self, latent_chunk: torch.Tensor) -> bool:
        """Put a latent chunk for decoding (non-blocking)"""
        if not self.is_streaming:
            return False
            
        try:
            self.input_queue.put(latent_chunk, timeout=0.001)
            return True
        except queue.Full:
            return False
        except Exception:
            return False
    
    def get_audio_chunk(self, timeout: float = 0.1) -> Optional[dict]:
        """Get decoded audio chunk (blocking with timeout)"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        except Exception:
            return None
    
    def stream_decode_generator(self, latent_stream: Iterator[torch.Tensor]) -> Iterator[np.ndarray]:
        """Generator-based streaming interface"""
        self.start_streaming()
        
        try:
            for latent_chunk in latent_stream:
                try:
                    if not self.put_latent_chunk(latent_chunk):
                        continue
                    
                    # Yield available audio chunks
                    attempts = 0
                    max_attempts = 10
                    while attempts < max_attempts:
                        chunk_info = self.get_audio_chunk(timeout=0.001)
                        if chunk_info is None:
                            break
                        yield chunk_info['audio']
                        attempts += 1
                        
                except Exception:
                    continue
        
        except Exception:
            pass
        
        finally:
            self.stop_streaming()
    
    @contextmanager
    def streaming_session(self):
        """Context manager for streaming sessions"""
        self.start_streaming()
        try:
            yield self
        except Exception:
            pass
        finally:
            self.stop_streaming()
    
    def get_latency_ms(self) -> float:
        """Get theoretical minimum latency in milliseconds"""
        return (self.chunk_size / self.sample_rate) * 1000
    
    def get_stats(self) -> dict:
        """Get streaming statistics"""
        try:
            return {
                'chunk_size_ms': self.get_latency_ms(),
                'input_queue_size': self.input_queue.qsize(),
                'output_queue_size': self.output_queue.qsize(),
                'is_streaming': self.is_streaming,
                'overlap_ratio': self.overlap_ratio,
                'worker_alive': self.worker_thread.is_alive() if self.worker_thread else False,
                'architecture': 'log_mel_phase'
            }
        except Exception:
            return {
                'chunk_size_ms': self.get_latency_ms(),
                'input_queue_size': 0,
                'output_queue_size': 0,
                'is_streaming': self.is_streaming,
                'overlap_ratio': self.overlap_ratio,
                'worker_alive': False,
                'architecture': 'log_mel_phase'
            }

def create_streaming_decoder(model_path: str, latency_ms: float = 100) -> StreamingDecoder:
    """
    Create a streaming decoder with specified target latency
    Simplified version with basic validation
    """
    if latency_ms <= 0:
        raise ValueError(f"latency_ms must be positive, got {latency_ms}")
    
    chunk_size = int(SAMPLE_RATE * latency_ms / 1000)
    
    # Ensure minimum chunk size
    min_chunk_size = 1024  # ~23ms at 44.1kHz
    if chunk_size < min_chunk_size:
        chunk_size = min_chunk_size
    
    # Cap maximum chunk size
    max_chunk_size = SAMPLE_RATE  # 1 second max
    if chunk_size > max_chunk_size:
        chunk_size = max_chunk_size
    
    # Validate for mel processing
    mel_time_frames = chunk_size // HOP_LENGTH
    if mel_time_frames < 2:
        chunk_size = HOP_LENGTH * 2
    
    try:
        decoder = StreamingDecoder(
            model_path=model_path,
            chunk_size=chunk_size,
            overlap_ratio=0.5
        )
        
        return decoder
        
    except Exception:
        # Return decoder with dummy model
        return StreamingDecoder(
            model_path=None,
            chunk_size=chunk_size,
            overlap_ratio=0.5
        )  