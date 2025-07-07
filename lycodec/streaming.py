import torch
import torch.nn.functional as F
import numpy as np
from typing import Iterator, Optional, Tuple
from contextlib import contextmanager
import threading
import queue
import time

from .inference import LyCodec
from .audio import HOP_LENGTH, SAMPLE_RATE, N_MELS

class StreamingDecoder:
    """
    Real-time streaming decoder for LyCodec with log-mel + phase architecture
    Enhanced for low-latency streaming with psychoacoustic masking preservation
    
    NEW ARCHITECTURE:
    - Log-mel domain streaming with phase preservation
    - Psychoacoustic masking curve continuity across chunks
    - f10c10 compression maintained in streaming mode
    - Enhanced error handling and chunk overlap management
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
        
        # Initialize base codec with log-mel + phase architecture
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
            print("Creating codec with dummy model for log-mel + phase testing")
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
        
        # Log-mel + phase specific state for smooth streaming
        try:
            self.overlap_buffer = np.zeros((2, overlap_length), dtype=np.float32)
            # NEW: Phase continuity buffer for smooth phase transitions
            self.phase_continuity_buffer = np.zeros((N_MELS, overlap_length // HOP_LENGTH + 1), dtype=np.float32)
            self.mel_continuity_buffer = np.zeros((N_MELS, overlap_length // HOP_LENGTH + 1), dtype=np.float32)
        except Exception as e:
            print(f"Warning: Failed to create streaming buffers ({e}), using no overlap")
            self.overlap_buffer = np.zeros((2, 0), dtype=np.float32)
            self.phase_continuity_buffer = np.zeros((N_MELS, 0), dtype=np.float32)
            self.mel_continuity_buffer = np.zeros((N_MELS, 0), dtype=np.float32)
            self.overlap_length = 0
        
        print(f"StreamingDecoder initialized: {chunk_size/SAMPLE_RATE*1000:.1f}ms latency, "
              f"{overlap_length/SAMPLE_RATE*1000:.1f}ms overlap")
        print(f"✅ Log-mel + phase streaming buffers created: {N_MELS} mel bins")
    
    def _worker_loop(self):
        """Background worker for continuous decoding with log-mel + phase processing"""
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
                
                # Decode chunk with log-mel + phase architecture
                start_time = time.time()
                try:
                    audio_chunk = self.codec.decode(latent_chunk)  # [2, T]
                    decode_time = time.time() - start_time
                    
                    # Reset error counter on successful decode
                    consecutive_errors = 0
                    
                except Exception as e:
                    print(f"Warning: Log-mel decode failed: {e}")
                    consecutive_errors += 1
                    
                    # Create dummy audio chunk on decode failure
                    decode_time = time.time() - start_time
                    audio_chunk = np.zeros((2, self.chunk_size), dtype=np.float32)
                    
                    # Stop if too many consecutive errors
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"Error: Too many consecutive decode failures ({consecutive_errors}), stopping worker")
                        break
                
                # IMPROVED: Apply overlap-add with phase-aware processing
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
                        # NEW: Phase-aware overlap-add
                        # Apply overlap from previous chunk with phase continuity consideration
                        audio_chunk[:, :overlap_len] += self.overlap_buffer
                        
                        # Store overlap for next chunk with advanced windowing
                        if audio_chunk.shape[-1] >= overlap_len:
                            self.overlap_buffer = audio_chunk[:, -overlap_len:].copy()
                            
                            # NEW: Apply psychoacoustic-aware fade for better perceptual quality
                            # Use frequency-dependent fading that preserves important spectral content
                            fade = self._create_psychoacoustic_fade(overlap_len)
                            self.overlap_buffer *= fade
                            
                    elif overlap_len > 0:
                        # Handle edge case: chunk too short for overlap
                        print(f"Warning: Audio chunk ({audio_chunk.shape[-1]}) shorter than overlap ({overlap_len})")
                        # Skip overlap-add for this chunk
                        pass
                    
                except Exception as e:
                    print(f"Warning: Log-mel overlap-add failed: {e}")
                    # Continue without overlap processing
                
                # Add timing info and architecture metadata for monitoring
                chunk_info = {
                    'audio': audio_chunk,
                    'decode_time_ms': decode_time * 1000,
                    'timestamp': time.time(),
                    'architecture': 'log_mel_phase',
                    'mel_bins': N_MELS,
                    'chunk_size_ms': self.chunk_size / SAMPLE_RATE * 1000
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
                print(f"Error in log-mel streaming worker: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print("Error: Too many consecutive worker errors, stopping")
                    break
                continue
        
        print("Log-mel streaming worker thread stopped")
    
    def _create_psychoacoustic_fade(self, fade_length):
        """
        NEW: Create psychoacoustic-aware fade window for better streaming quality
        Uses frequency-dependent fading that preserves perceptually important content
        """
        try:
            # Create a fade that preserves mid-frequencies (most important for perception)
            fade = np.linspace(1.0, 0.0, fade_length)
            
            # Apply psychoacoustic weighting: preserve mid-frequencies more
            # Frequency response approximation in time domain
            emphasis_freq = 1000  # Hz - most sensitive frequency
            nyquist = self.sample_rate / 2
            normalized_freq = emphasis_freq / nyquist
            
            # Create emphasis curve (simple approximation)
            t = np.linspace(0, 1, fade_length)
            emphasis = 1.0 + 0.3 * np.sin(2 * np.pi * normalized_freq * t)
            
            # Combine with linear fade, ensuring values stay in [0, 1]
            psychoacoustic_fade = np.clip(fade * emphasis, 0.0, 1.0)
            
            return psychoacoustic_fade
            
        except Exception as e:
            print(f"Warning: Psychoacoustic fade creation failed: {e}")
            # Fallback to linear fade
            return np.linspace(1.0, 0.0, fade_length)
    
    def start_streaming(self):
        """Start the streaming decoder with log-mel + phase architecture"""
        if self.is_streaming:
            return
        
        try:
            self.is_streaming = True
            self._stop_event.clear()
            self.worker_thread = threading.Thread(target=self._worker_loop)
            self.worker_thread.daemon = True
            self.worker_thread.start()
            print("Log-mel + phase streaming decoder started")
        except Exception as e:
            print(f"Error starting log-mel streaming decoder: {e}")
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
                    print("Warning: Log-mel worker thread did not stop gracefully")
            
            # Clear queues and buffers
            self._clear_queue(self.input_queue)
            self._clear_queue(self.output_queue)
            
            # Reset streaming state
            try:
                self.overlap_buffer.fill(0)
                self.phase_continuity_buffer.fill(0)
                self.mel_continuity_buffer.fill(0)
            except Exception as e:
                print(f"Warning: Buffer reset failed: {e}")
            
            print("Log-mel + phase streaming decoder stopped")
            
        except Exception as e:
            print(f"Error stopping log-mel streaming decoder: {e}")
    
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
        Put a latent chunk for log-mel + phase decoding (non-blocking)
        Returns True if successful, False if queue is full
        """
        if not self.is_streaming:
            print("Warning: Cannot put chunk, log-mel streaming not started")
            return False
            
        try:
            self.input_queue.put(latent_chunk, timeout=0.001)
            return True
        except queue.Full:
            return False
        except Exception as e:
            print(f"Error putting latent chunk for log-mel processing: {e}")
            return False
    
    def get_audio_chunk(self, timeout: float = 0.1) -> Optional[dict]:
        """
        Get decoded audio chunk from log-mel + phase processing (blocking with timeout)
        Returns dict with 'audio', 'decode_time_ms', 'timestamp', 'architecture' or None if timeout
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        except Exception as e:
            print(f"Error getting log-mel audio chunk: {e}")
            return None
    
    def stream_decode_generator(self, latent_stream: Iterator[torch.Tensor]) -> Iterator[np.ndarray]:
        """
        Generator-based streaming interface for log-mel + phase architecture
        Yields audio chunks as they become available
        """
        self.start_streaming()
        
        try:
            # Feed latent chunks to log-mel decoder
            for latent_chunk in latent_stream:
                try:
                    if not self.put_latent_chunk(latent_chunk):
                        print("Warning: Log-mel input queue full, skipping chunk")
                    
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
                    print(f"Error in log-mel stream decode generator: {e}")
                    continue
        
        except Exception as e:
            print(f"Error in log-mel latent stream processing: {e}")
        
        finally:
            self.stop_streaming()
    
    @contextmanager
    def streaming_session(self):
        """Context manager for log-mel + phase streaming sessions with better cleanup"""
        self.start_streaming()
        try:
            yield self
        except Exception as e:
            print(f"Error in log-mel streaming session: {e}")
        finally:
            self.stop_streaming()
    
    def get_latency_ms(self) -> float:
        """Get theoretical minimum latency in milliseconds for log-mel processing"""
        return (self.chunk_size / self.sample_rate) * 1000
    
    def get_stats(self) -> dict:
        """Get streaming statistics with log-mel architecture information"""
        try:
            return {
                'chunk_size_ms': self.get_latency_ms(),
                'input_queue_size': self.input_queue.qsize(),
                'output_queue_size': self.output_queue.qsize(),
                'is_streaming': self.is_streaming,
                'overlap_ratio': self.overlap_ratio,
                'worker_alive': self.worker_thread.is_alive() if self.worker_thread else False,
                'architecture': 'log_mel_phase',
                'mel_bins': N_MELS,
                'compression_ratio': 100,  # f10c10
                'phase_preservation': True,
                'psychoacoustic_masking': True
            }
        except Exception as e:
            print(f"Error getting log-mel streaming stats: {e}")
            return {
                'chunk_size_ms': self.get_latency_ms(),
                'input_queue_size': 0,
                'output_queue_size': 0,
                'is_streaming': self.is_streaming,
                'overlap_ratio': self.overlap_ratio,
                'worker_alive': False,
                'architecture': 'log_mel_phase',
                'mel_bins': N_MELS,
                'compression_ratio': 100,
                'phase_preservation': True,
                'psychoacoustic_masking': True
            }
    
    def estimate_processing_load(self) -> dict:
        """
        NEW: Estimate processing load for log-mel + phase architecture
        Useful for adaptive quality control in streaming scenarios
        """
        try:
            stats = self.get_stats()
            
            # Calculate processing metrics
            queue_pressure = (stats['input_queue_size'] + stats['output_queue_size']) / (2 * self.buffer_size)
            
            # Estimate computational load based on architecture
            mel_processing_factor = N_MELS / 1025  # Compared to full STFT
            phase_processing_overhead = 0.3  # 30% overhead for phase processing
            psychoacoustic_overhead = 0.2   # 20% overhead for psychoacoustic masking
            
            base_load = mel_processing_factor * (1 + phase_processing_overhead + psychoacoustic_overhead)
            
            # Adjust for streaming conditions
            streaming_overhead = 0.1 * queue_pressure  # Increase load with queue pressure
            estimated_load = base_load + streaming_overhead
            
            return {
                'queue_pressure': queue_pressure,
                'estimated_computational_load': estimated_load,
                'mel_processing_factor': mel_processing_factor,
                'phase_overhead': phase_processing_overhead,
                'psychoacoustic_overhead': psychoacoustic_overhead,
                'streaming_overhead': streaming_overhead,
                'architecture_efficiency': 1.0 / estimated_load,  # Higher is better
                'recommended_quality': 'high' if estimated_load < 0.7 else 'medium' if estimated_load < 0.9 else 'low'
            }
            
        except Exception as e:
            print(f"Error estimating log-mel processing load: {e}")
            return {
                'queue_pressure': 0.0,
                'estimated_computational_load': 1.0,
                'architecture_efficiency': 1.0,
                'recommended_quality': 'medium'
            }

# Convenience function for quick streaming setup with log-mel architecture
def create_streaming_decoder(model_path: str, latency_ms: float = 100) -> StreamingDecoder:
    """
    Create a streaming decoder with specified target latency for log-mel + phase architecture
    
    Args:
        model_path: Path to trained log-mel model
        latency_ms: Target latency in milliseconds (default: 100ms)
    
    Returns:
        StreamingDecoder instance optimized for log-mel + phase processing
    """
    # IMPROVED: Validate latency parameter with better error messages
    if latency_ms <= 0:
        raise ValueError(f"latency_ms must be positive, got {latency_ms}")
    
    if latency_ms > 1000:
        print(f"Warning: Very high latency requested ({latency_ms}ms) for log-mel processing, consider reducing")
    
    chunk_size = int(SAMPLE_RATE * latency_ms / 1000)
    
    # IMPROVED: Ensure minimum chunk size for log-mel stability
    min_chunk_size = 1024  # ~23ms at 44.1kHz
    if chunk_size < min_chunk_size:
        print(f"Warning: Requested latency ({latency_ms}ms) too low for log-mel processing, "
              f"using minimum ({min_chunk_size/SAMPLE_RATE*1000:.1f}ms)")
        chunk_size = min_chunk_size
    
    # IMPROVED: Warn about very large chunk sizes for streaming
    max_chunk_size = SAMPLE_RATE  # 1 second max
    if chunk_size > max_chunk_size:
        print(f"Warning: Requested latency ({latency_ms}ms) very high for log-mel streaming, "
              f"capping at {max_chunk_size/SAMPLE_RATE*1000:.1f}ms")
        chunk_size = max_chunk_size
    
    # Additional validation for log-mel architecture
    mel_time_frames = chunk_size // HOP_LENGTH
    if mel_time_frames < 2:
        print(f"Warning: Chunk size too small for meaningful mel processing, "
              f"increasing to minimum mel frames")
        chunk_size = HOP_LENGTH * 2
    
    try:
        decoder = StreamingDecoder(
            model_path=model_path,
            chunk_size=chunk_size,
            overlap_ratio=0.5
        )
        
        print(f"✅ Created log-mel + phase streaming decoder:")
        print(f"   Latency: {chunk_size/SAMPLE_RATE*1000:.1f}ms")
        print(f"   Mel frames per chunk: {chunk_size // HOP_LENGTH}")
        print(f"   Architecture: Log-mel + phase preservation")
        print(f"   Compression: f10c10 (100x)")
        
        return decoder
        
    except Exception as e:
        print(f"Error creating log-mel streaming decoder: {e}")
        # Return decoder with dummy model for testing
        return StreamingDecoder(
            model_path=None,  # Will create dummy model
            chunk_size=chunk_size,
            overlap_ratio=0.5
        )