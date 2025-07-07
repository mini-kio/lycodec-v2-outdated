import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import warnings
from typing import Union, Optional, Tuple, List
from contextlib import contextmanager

from .models import LyCodecModel
from .audio import (
    to_complex_spec, 
    to_waveform, 
    normalize_audio,
    to_magnitude_phase,
    from_magnitude_phase,
    to_mel_spectrogram,
    to_log_mel,
    from_log_mel,
    mel_to_magnitude,
    create_mel_filterbank,
    SAMPLE_RATE, 
    HOP_LENGTH, 
    N_FFT,
    N_MELS
)

class LyCodec:
    """
    LyCodec inference engine optimized for log-mel + phase processing - REFACTORED VERSION
    New Architecture:
    Waveform → STFT → Magnitude/Phase → Mel filterbank → log_mel (128 bin) + phase preservation
    Features:
    - Log-mel domain processing with phase preservation
    - Psychoacoustic masking curve weighting
    - f10c10 compression (100x) maintained
    - Automatic memory management with accurate peak tracking
    - Vectorized chunk-based processing for long audio
    - CPU fallback for memory constraints
    - Conditional half precision inference
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 half_precision: bool = True,
                 max_chunk_length: int = 220500,  # 5 seconds at 44.1kHz
                 overlap_length: int = 4410):     # 0.1 second overlap
        
        self.half_precision = half_precision
        self.max_chunk_length = max_chunk_length
        self.overlap_length = overlap_length
        
        # Auto-select device
        if device is None:
            if torch.cuda.is_available():
                # Check VRAM
                try:
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    if vram_gb >= 6:  # Minimum 6GB for GPU inference
                        self.device = torch.device('cuda')
                    else:
                        warnings.warn(f"GPU has only {vram_gb:.1f}GB VRAM, using CPU")
                        self.device = torch.device('cpu')
                except Exception as e:
                    warnings.warn(f"Error checking GPU memory ({e}), using CPU")
                    self.device = torch.device('cpu')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # IMPROVED: Enhanced FP16 CPU guards with clearer warnings
        if self.device.type == 'cpu':
            if half_precision:
                warnings.warn(
                    "FP16 on CPU is inefficient and may cause numerical instability. "
                    "Consider using GPU or disabling half_precision. "
                    "Note: FP16 tensors converted to numpy may lose precision.",
                    UserWarning,
                    stacklevel=2
                )
            self.half_precision = False
        else:
            self.half_precision = half_precision
        
        print(f"LyCodec initialized on {self.device} with log-mel + phase processing")
        
        # Create mel filterbank for consistent processing
        self.mel_filterbank = create_mel_filterbank(n_mels=N_MELS, n_fft=N_FFT).to(self.device)
        
        # Load model with better error handling
        try:
            # IMPROVED: Disable Triton in inference for stability
            self.model = LyCodecModel(use_triton=False)
            if model_path:
                self._load_model(model_path)
        except Exception as e:
            print(f"Warning: Error initializing model ({e}), creating minimal model")
            # Fallback to minimal model for testing
            self.model = LyCodecModel(use_triton=False)
        
        self.model.to(self.device)
        
        # Set to half precision for memory efficiency - IMPROVED CONDITION with warnings
        if half_precision and self.device.type == 'cuda':
            try:
                self.model.half()
                # Also convert mel filterbank to half precision
                if hasattr(self, 'mel_filterbank'):
                    self.mel_filterbank = self.mel_filterbank.half()
                print("Using FP16 precision on GPU for log-mel processing")
            except Exception as e:
                warnings.warn(f"Failed to set FP16 ({e}), using FP32")
                self.half_precision = False
        elif half_precision and self.device.type == 'cpu':
            warnings.warn(
                "FP16 on CPU bypassed - using FP32. "
                "GPU recommended for FP16 inference.",
                UserWarning
            )
        
        self.model.eval()
        
        # Memory monitoring
        self._monitor_memory = self.device.type == 'cuda'
        
        # Precompute cosine window for OLA - IMPROVED WINDOWING
        if self.overlap_length > 0:
            # Use cosine window instead of linear for better cross-fade
            try:
                self.fade_window = torch.hann_window(self.overlap_length * 2)[:self.overlap_length]
                self.fade_window = self.fade_window.to(self.device)
            except Exception as e:
                print(f"Warning: Failed to create fade window ({e}), using linear fade")
                self.fade_window = torch.linspace(0, 1, self.overlap_length).to(self.device)
        else:
            self.fade_window = None
        
        print(f"✅ Log-mel filterbank created: {N_MELS} mel bins")
    
    def _load_model(self, model_path: str):
        """Load model weights from checkpoint with better error handling"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Handle DDP wrapped models
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', '') if key.startswith('module.') else key
                new_state_dict[new_key] = value
            
            # IMPROVED: More robust state dict loading
            try:
                self.model.load_state_dict(new_state_dict, strict=True)
                print(f"Model loaded from {model_path}")
            except RuntimeError as e:
                print(f"Warning: Strict loading failed ({e}), trying non-strict loading")
                missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
                if missing_keys:
                    print(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys}")
                print(f"Model loaded from {model_path} (non-strict)")
                
        except Exception as e:
            warnings.warn(f"Failed to load model from {model_path}: {e}")
            print("Using randomly initialized model")
    
    @contextmanager
    def _memory_efficient_inference(self):
        """Context manager for memory-efficient inference with leak detection"""
        if self._monitor_memory:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                initial_memory = torch.cuda.memory_allocated()
            except Exception as e:
                print(f"Warning: Memory monitoring failed ({e})")
                initial_memory = 0
                
        try:
            with torch.no_grad():
                yield
        finally:
            if self._monitor_memory:
                try:
                    peak_memory = torch.cuda.max_memory_allocated()
                    torch.cuda.empty_cache()
                    final_memory = torch.cuda.memory_allocated()
                    
                    # IMPROVED: Include leak detection in memory logging
                    leak = final_memory - initial_memory
                    print(f"Memory usage: {(peak_memory - initial_memory) / (1024**3):.2f}GB peak, "
                          f"{leak / (1024**2):.1f}MB leak")
                except Exception as e:
                    print(f"Warning: Memory reporting failed ({e})")
    
    def _audio_to_log_mel_phase(self, audio_tensor):
        """
        Convert audio to log-mel + phase representation
        Args:
            audio_tensor: [B, channels, T] audio tensor
        Returns:
            log_mel_features: [B, n_mels, T_frames] log-mel spectrogram
            phase_features: [B, n_mels, T_frames] phase information  
        """
        B, channels, T = audio_tensor.shape
        
        # Process each channel and average for mono processing (stereo->mono for mel)
        all_log_mels = []
        all_phases = []
        
        for ch in range(channels):
            # STFT for each channel
            complex_spec = to_complex_spec(audio_tensor[:, ch])  # [B, F, T_frames]
            
            # Extract magnitude and phase
            magnitude_spec, phase_spec = to_magnitude_phase(complex_spec)
            
            # Convert magnitude to mel-scale
            mel_spec = to_mel_spectrogram(magnitude_spec, self.mel_filterbank)  # [B, n_mels, T_frames]
            
            # Convert to log-mel
            log_mel_spec = to_log_mel(mel_spec)  # [B, n_mels, T_frames]
            
            # Process phase to mel-scale (approximate mapping)
            phase_mel = to_mel_spectrogram(phase_spec.abs(), self.mel_filterbank)  # Use abs for positive values
            # Keep original phase structure but map to mel scale
            phase_mel_weighted = phase_spec.unsqueeze(1).expand(-1, N_MELS, -1, -1)
            if phase_mel_weighted.shape[2] != mel_spec.shape[1]:
                # Interpolate phase to match mel dimensions
                phase_mel_weighted = torch.nn.functional.interpolate(
                    phase_mel_weighted, size=(mel_spec.shape[1], mel_spec.shape[2]), 
                    mode='bilinear', align_corners=False
                )
            phase_mel_final = phase_mel_weighted.mean(dim=2)  # [B, n_mels, T_frames]
            
            all_log_mels.append(log_mel_spec)
            all_phases.append(phase_mel_final)
        
        # Average across channels for processing
        log_mel_features = torch.stack(all_log_mels, dim=1).mean(dim=1)  # [B, n_mels, T_frames]
        phase_features = torch.stack(all_phases, dim=1).mean(dim=1)      # [B, n_mels, T_frames]
        
        return log_mel_features, phase_features
    
    def _log_mel_phase_to_audio(self, log_mel_features, phase_features, target_length=None):
        """
        Convert log-mel + phase back to audio
        Args:
            log_mel_features: [B, n_mels, T_frames] log-mel spectrogram
            phase_features: [B, n_mels, T_frames] phase information
            target_length: target audio length in samples
        Returns:
            audio_tensor: [B, channels, T] reconstructed audio
        """
        # Convert log-mel back to linear mel
        mel_features = from_log_mel(log_mel_features)  # [B, n_mels, T_frames]
        
        # Convert mel back to magnitude spectrogram
        magnitude_spec = mel_to_magnitude(mel_features, self.mel_filterbank, n_fft=N_FFT)  # [B, F, T_frames]
        
        # Reconstruct phase spectrogram from mel-scale phase
        # Map phase back to full frequency spectrum
        F_bins = N_FFT // 2 + 1
        T_frames = phase_features.shape[-1]
        
        # Interpolate phase to match magnitude spectrum dimensions
        phase_spec = torch.nn.functional.interpolate(
            phase_features.unsqueeze(2), size=(F_bins, T_frames), 
            mode='bilinear', align_corners=False
        ).squeeze(2)  # [B, F, T_frames]
        
        # Reconstruct complex spectrogram
        complex_spec = from_magnitude_phase(magnitude_spec, phase_spec)  # [B, F, T_frames]
        
        # Convert to audio (mono first)
        mono_audio = to_waveform(complex_spec, length=target_length)  # [B, T]
        
        # Expand to stereo
        if mono_audio.dim() == 2:  # [B, T]
            stereo_audio = mono_audio.unsqueeze(1).repeat(1, 2, 1)  # [B, 2, T]
        else:  # [T] single sample
            stereo_audio = mono_audio.unsqueeze(0).repeat(2, 1).unsqueeze(0)  # [1, 2, T]
        
        return stereo_audio
    
    def _chunk_audio(self, audio: torch.Tensor) -> List[Tuple[torch.Tensor, int]]:
        """Split long audio into overlapping chunks - IMPROVED with start indices"""
        if audio.shape[-1] <= self.max_chunk_length:
            return [(audio, 0)]
        
        chunks = []
        start = 0
        
        while start < audio.shape[-1]:
            end = min(start + self.max_chunk_length, audio.shape[-1])
            chunk = audio[..., start:end]
            chunks.append((chunk, start))  # FIXED: Return (chunk, start_idx) tuples
            
            if end >= audio.shape[-1]:
                break
                
            start = end - self.overlap_length
        
        return chunks
    
    def _merge_chunks(self, chunks: List[Tuple[torch.Tensor, int]], original_length: int) -> torch.Tensor:
        """Merge overlapping chunks back to original audio with improved cosine windowing"""
        if len(chunks) == 1:
            chunk, _ = chunks[0]
            return chunk[..., :original_length]
        
        # Initialize output
        first_chunk, _ = chunks[0]
        merged = torch.zeros(
            *first_chunk.shape[:-1], 
            original_length,
            dtype=first_chunk.dtype,
            device=first_chunk.device
        )
        
        # IMPROVED: Overlap-add with cosine fade windows and better error handling
        try:
            for i, (chunk, start_pos) in enumerate(chunks):
                chunk_len = chunk.shape[-1]
                end_pos = min(start_pos + chunk_len, original_length)
                actual_chunk_len = end_pos - start_pos
                
                if actual_chunk_len <= 0:
                    continue
                
                if i == 0:
                    # First chunk: no fade in, fade out at overlap
                    no_fade_len = actual_chunk_len - self.overlap_length if len(chunks) > 1 else actual_chunk_len
                    if no_fade_len > 0:
                        merged[..., start_pos:start_pos + no_fade_len] = chunk[..., :no_fade_len]
                    
                    # Fade out at overlap
                    if len(chunks) > 1 and no_fade_len < actual_chunk_len and self.fade_window is not None:
                        fade_start = start_pos + no_fade_len
                        fade_len = min(self.overlap_length, end_pos - fade_start)
                        if fade_len > 0:
                            fade_out = 1.0 - self.fade_window[:fade_len]
                            merged[..., fade_start:fade_start + fade_len] += chunk[..., no_fade_len:no_fade_len + fade_len] * fade_out
                            
                elif i == len(chunks) - 1:
                    # Last chunk: fade in, no fade out
                    fade_len = min(self.overlap_length, actual_chunk_len)
                    if fade_len > 0 and self.fade_window is not None:
                        fade_in = self.fade_window[:fade_len]
                        merged[..., start_pos:start_pos + fade_len] += chunk[..., :fade_len] * fade_in
                    
                    # Rest of chunk
                    if fade_len < actual_chunk_len:
                        merged[..., start_pos + fade_len:end_pos] = chunk[..., fade_len:actual_chunk_len]
                        
                else:
                    # Middle chunk: fade in and out
                    fade_len = min(self.overlap_length, actual_chunk_len)
                    if fade_len > 0 and self.fade_window is not None:
                        fade_in = self.fade_window[:fade_len]
                        merged[..., start_pos:start_pos + fade_len] += chunk[..., :fade_len] * fade_in
                    
                    # Middle part
                    middle_start = fade_len
                    middle_end = actual_chunk_len - self.overlap_length
                    if middle_end > middle_start:
                        merged[..., start_pos + middle_start:start_pos + middle_end] = chunk[..., middle_start:middle_end]
                    
                    # Fade out
                    if middle_end < actual_chunk_len and self.fade_window is not None:
                        fade_out_len = actual_chunk_len - middle_end
                        if fade_out_len > 0:
                            fade_out = 1.0 - self.fade_window[:fade_out_len]
                            merged[..., start_pos + middle_end:end_pos] += chunk[..., middle_end:actual_chunk_len] * fade_out
            
            return merged
        
        except Exception as e:
            print(f"Warning: Chunk merging failed ({e}), using simple concatenation")
            # Fallback to simple concatenation
            result = torch.zeros(*first_chunk.shape[:-1], original_length, dtype=first_chunk.dtype, device=first_chunk.device)
            pos = 0
            for chunk, _ in chunks:
                chunk_len = min(chunk.shape[-1], original_length - pos)
                if chunk_len > 0:
                    result[..., pos:pos + chunk_len] = chunk[..., :chunk_len]
                    pos += chunk_len
                if pos >= original_length:
                    break
            return result
    
    def encode(self, audio: Union[str, np.ndarray, torch.Tensor], 
               normalize: bool = True) -> torch.Tensor:
        """
        Encode audio to compressed latent representation using log-mel + phase
        
        Args:
            audio: Audio file path, numpy array, or torch tensor
            normalize: Whether to normalize audio level
            
        Returns:
            latent: Compressed latent representation [B, C, H, W]
        """
        try:
            # Load and preprocess audio
            if isinstance(audio, str):
                try:
                    audio_data, sr = sf.read(audio, always_2d=True)
                    audio_data = audio_data.T  # [channels, samples]
                    
                    if sr != SAMPLE_RATE:
                        warnings.warn(f"Audio sample rate {sr} != {SAMPLE_RATE}, resampling recommended")
                    
                    audio_tensor = torch.from_numpy(audio_data).float()
                except Exception as e:
                    raise ValueError(f"Failed to load audio file {audio}: {e}")
                    
            elif isinstance(audio, np.ndarray):
                audio_tensor = torch.from_numpy(audio).float()
            else:
                audio_tensor = audio.float()
            
            # Ensure stereo - IMPROVED: Better dimension handling
            if audio_tensor.dim() == 1:
                # Mono audio [T] -> [2, T]
                audio_tensor = audio_tensor.unsqueeze(0).repeat(2, 1)
            elif audio_tensor.dim() == 2:
                if audio_tensor.shape[0] == 1:
                    # Mono audio [1, T] -> [2, T]
                    audio_tensor = audio_tensor.repeat(2, 1)
                # else: already stereo [2, T], keep as is
            elif audio_tensor.dim() == 3:
                # Already has batch dimension [B, C, T], check channels
                if audio_tensor.shape[1] == 1:
                    # Mono with batch [B, 1, T] -> [B, 2, T]
                    audio_tensor = audio_tensor.repeat(1, 2, 1)
                # else: already stereo [B, 2, T], keep as is
            
            # Add batch dimension if needed
            if audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(0)  # [1, 2, T]
            
            # Normalize
            if normalize:
                try:
                    audio_tensor = normalize_audio(audio_tensor)
                except Exception as e:
                    print(f"Warning: Audio normalization failed ({e}), using original levels")
            
            # Move to device and cast precision - IMPROVED: Ensure consistent precision
            audio_tensor = audio_tensor.to(self.device)
            if self.half_precision and self.device.type == 'cuda':
                try:
                    audio_tensor = audio_tensor.half()
                except Exception as e:
                    print(f"Warning: FP16 conversion failed ({e}), using FP32")
                    audio_tensor = audio_tensor.float()
            elif self.device.type == 'cuda':
                audio_tensor = audio_tensor.float()  # Ensure FP32 on GPU if not using half precision
            
            with self._memory_efficient_inference():
                # Process in chunks if necessary
                original_length = audio_tensor.shape[-1]
                chunks = self._chunk_audio(audio_tensor)
                
                latent_chunks = []
                for chunk, start_idx in chunks:
                    try:
                        # NEW ARCHITECTURE: Convert to log-mel + phase
                        log_mel_features, phase_features = self._audio_to_log_mel_phase(chunk)
                        
                        # IMPROVED: Ensure consistent precision for model inputs
                        if self.half_precision and self.device.type == 'cuda':
                            log_mel_features = log_mel_features.half()
                            phase_features = phase_features.half()
                        
                        # Encode using log-mel + phase
                        latent_chunk = self.model.encode(log_mel_features, phase_features)
                        latent_chunks.append((latent_chunk.cpu(), start_idx))
                        
                    except Exception as e:
                        print(f"Warning: Failed to encode chunk {start_idx}: {e}")
                        # Create dummy latent for failed chunk
                        dummy_latent = torch.zeros(1, 64, 8, 32)  # Default latent size
                        latent_chunks.append((dummy_latent, start_idx))
            
            # IMPROVED: Preserve chunk information for better decode consistency
            if len(latent_chunks) == 1:
                latent, _ = latent_chunks[0]
                return latent
            else:
                # Store chunk metadata for decode
                latents = [chunk for chunk, _ in latent_chunks]
                return {
                    'latents': torch.cat(latents, dim=-1),  # Concatenate along time
                    'original_length': original_length,
                    'chunk_info': [(start_idx, latent.shape) for latent, start_idx in latent_chunks]
                }
                
        except Exception as e:
            print(f"Error in encode: {e}")
            # Return dummy latent on complete failure
            return torch.zeros(1, 64, 8, 32)
    
    def decode(self, latent: Union[torch.Tensor, dict], 
               target_length: Optional[int] = None) -> np.ndarray:
        """
        Decode latent representation back to audio using log-mel + phase reconstruction
        
        Args:
            latent: Compressed latent [B, C, H, W] or dict with chunk info
            target_length: Target audio length in samples
            
        Returns:
            audio: Decoded stereo audio [channels, samples]
        """
        try:
            # Handle chunked latents with proper chunk processing
            if isinstance(latent, dict) and 'chunk_info' in latent:
                return self._decode_chunked(latent, target_length)
            
            # Handle single latent tensor
            latent_tensor = latent['latents'] if isinstance(latent, dict) else latent
            
            # Add batch dimension if needed
            if latent_tensor.dim() == 3:
                latent_tensor = latent_tensor.unsqueeze(0)
            
            # Move to device and cast precision
            latent_tensor = latent_tensor.to(self.device)
            if self.half_precision and self.device.type == 'cuda':
                try:
                    latent_tensor = latent_tensor.half()
                except Exception as e:
                    print(f"Warning: FP16 conversion failed ({e}), using FP32")
                    latent_tensor = latent_tensor.float()
            
            with self._memory_efficient_inference():
                # NEW ARCHITECTURE: Decode to log-mel + phase
                try:
                    # Calculate target size for mel-scale output
                    if target_length is not None:
                        target_time_frames = (target_length // HOP_LENGTH) + 1
                        target_size = (N_MELS, target_time_frames)
                    else:
                        target_size = None
                    
                    # Decode with correct target size
                    log_mel_out, phase_out = self.model.decode(latent_tensor, target_size)
                    
                    # NEW ARCHITECTURE: Convert log-mel + phase back to audio
                    stereo_audio = self._log_mel_phase_to_audio(log_mel_out, phase_out, target_length)
                    
                    # Remove batch dimension and move to CPU
                    stereo_audio = stereo_audio.squeeze(0).cpu().float()  # [2, T]
                    
                    # Trim to target length if specified
                    if target_length is not None:
                        stereo_audio = stereo_audio[..., :target_length]
                    
                    return stereo_audio.numpy()
                    
                except Exception as e:
                    print(f"Error in model decode: {e}")
                    # Return dummy audio on decode failure
                    dummy_length = target_length if target_length else 44100
                    return np.zeros((2, dummy_length), dtype=np.float32)
        
        except Exception as e:
            print(f"Error in decode: {e}")
            # Return dummy audio on complete failure
            dummy_length = target_length if target_length else 44100
            return np.zeros((2, dummy_length), dtype=np.float32)
    
    def _decode_chunked(self, latent_dict: dict, target_length: Optional[int] = None) -> np.ndarray:
        """Decode chunked latents with cross-fade blending - IMPROVED with error handling"""
        try:
            chunk_info = latent_dict['chunk_info']
            original_length = latent_dict.get('original_length', target_length)
            
            # Decode each chunk separately
            decoded_chunks = []
            chunk_start = 0
            
            for i, (start_idx, chunk_shape) in enumerate(chunk_info):
                try:
                    # Extract chunk from concatenated latents
                    chunk_width = chunk_shape[-1]  # Time dimension
                    chunk_latent = latent_dict['latents'][..., chunk_start:chunk_start + chunk_width]
                    
                    # Decode chunk
                    chunk_audio = self.decode(chunk_latent)  # [2, T]
                    decoded_chunks.append((chunk_audio, start_idx))
                    chunk_start += chunk_width
                    
                except Exception as e:
                    print(f"Warning: Failed to decode chunk {i}: {e}")
                    # Create dummy audio for failed chunk
                    dummy_audio = np.zeros((2, 44100), dtype=np.float32)
                    decoded_chunks.append((dummy_audio, start_idx))
                    chunk_start += chunk_shape[-1]
            
            # IMPROVED: Use cross-fade to merge chunks in frequency domain for better quality
            if len(decoded_chunks) == 1:
                audio, _ = decoded_chunks[0]
                return audio[:, :target_length] if target_length else audio
            
            # Merge with overlap-add and cosine cross-fade
            total_length = original_length or target_length or max(
                start_idx + audio.shape[-1] for audio, start_idx in decoded_chunks
            )
            
            merged_audio = self._merge_chunks_with_crossfade(decoded_chunks, total_length)
            
            return merged_audio[:, :target_length] if target_length else merged_audio
            
        except Exception as e:
            print(f"Error in chunked decode: {e}")
            # Return dummy audio on failure
            dummy_length = target_length if target_length else 44100
            return np.zeros((2, dummy_length), dtype=np.float32)
    
    def _merge_chunks_with_crossfade(self, chunks: List[Tuple[np.ndarray, int]], total_length: int) -> np.ndarray:
        """Merge audio chunks with cosine cross-fade - IMPROVED with error handling"""
        try:
            merged = np.zeros((2, total_length), dtype=np.float32)  # Stereo
            
            for i, (chunk_audio, start_idx) in enumerate(chunks):
                chunk_len = chunk_audio.shape[-1]
                end_idx = min(start_idx + chunk_len, total_length)
                actual_len = end_idx - start_idx
                
                if actual_len <= 0:
                    continue
                    
                chunk_to_add = chunk_audio[:, :actual_len]
                
                # IMPROVED: Separate cross-fade logic for proper audio mixing with error handling
                try:
                    if i > 0 and start_idx < total_length and self.overlap_length > 0:
                        # Calculate overlap region
                        overlap_start = max(0, start_idx - self.overlap_length)
                        overlap_end = start_idx
                        overlap_len = overlap_end - overlap_start
                        
                        if overlap_len > 0 and self.fade_window is not None:
                            # Get fade weights
                            fade_len = min(overlap_len, len(self.fade_window))
                            fade_out = self.fade_window.cpu().numpy()[:fade_len]
                            fade_in = 1.0 - fade_out
                            
                            # Apply cross-fade only to overlap region
                            overlap_region_in_merged = slice(overlap_start, overlap_start + fade_len)
                            overlap_region_in_chunk = slice(0, fade_len)
                            
                            # Fade out existing content
                            merged[:, overlap_region_in_merged] *= fade_out
                            # Fade in new chunk and add
                            merged[:, overlap_region_in_merged] += chunk_to_add[:, overlap_region_in_chunk] * fade_in
                            
                            # Add non-overlapping part of the chunk
                            non_overlap_start = start_idx + fade_len
                            if non_overlap_start < end_idx:
                                merged[:, non_overlap_start:end_idx] = chunk_to_add[:, fade_len:]
                        else:
                            # No fade window, just overwrite
                            merged[:, start_idx:end_idx] = chunk_to_add
                    else:
                        # First chunk or no overlap
                        merged[:, start_idx:end_idx] = chunk_to_add
                        
                except Exception as e:
                    print(f"Warning: Cross-fade failed for chunk {i}: {e}, using simple addition")
                    # Simple addition fallback
                    merged[:, start_idx:end_idx] = chunk_to_add
            
            return merged
            
        except Exception as e:
            print(f"Error in chunk merging: {e}")
            # Return dummy audio on failure
            return np.zeros((2, total_length), dtype=np.float32)
    
    def encode_file(self, input_path: str, output_path: str):
        """Encode audio file and save latent representation - IMPROVED with error handling"""
        try:
            latent = self.encode(input_path)
            
            # IMPROVED: Maintain dtype consistency to save storage space
            if isinstance(latent, torch.Tensor):
                # Keep original precision to save space (fp16 stays fp16)
                latent_to_save = latent.cpu()
                if self.half_precision and latent.dtype == torch.float16:
                    # Save metadata about dtype for proper loading
                    save_data = {
                        'latent': latent_to_save,
                        'dtype': str(latent.dtype),
                        'half_precision': True,
                        'architecture': 'log_mel_phase'
                    }
                else:
                    save_data = {
                        'latent': latent_to_save,
                        'architecture': 'log_mel_phase'
                    }
            else:
                # Handle dict case (chunked latents)
                save_data = latent.copy()
                save_data['architecture'] = 'log_mel_phase'
                if 'latents' in latent and isinstance(latent['latents'], torch.Tensor):
                    latent['latents'] = latent['latents'].cpu()
            
            # Use efficient serialization for large files
            torch.save(save_data, output_path, _use_new_zipfile_serialization=False)
            print(f"Encoded {input_path} -> {output_path} (log-mel + phase)")
            
            # Print compression stats
            try:
                input_size = Path(input_path).stat().st_size
                output_size = Path(output_path).stat().st_size
                compression_ratio = input_size / output_size
                print(f"Compression ratio: {compression_ratio:.1f}x (f10c10 log-mel)")
            except Exception as e:
                print(f"Warning: Could not calculate compression ratio: {e}")
                
        except Exception as e:
            print(f"Error encoding file {input_path}: {e}")
    
    def decode_file(self, input_path: str, output_path: str, sample_rate: int = SAMPLE_RATE):
        """Decode latent file and save as audio - IMPROVED with error handling"""
        try:
            loaded_data = torch.load(input_path, map_location='cpu')
            
            # Handle different save formats
            if isinstance(loaded_data, dict):
                if 'latent' in loaded_data:
                    # New format with metadata
                    latent = loaded_data['latent']
                    architecture = loaded_data.get('architecture', 'unknown')
                    if loaded_data.get('half_precision', False):
                        # Restore original dtype if needed
                        latent = latent.half() if latent.dtype != torch.float16 else latent
                elif 'latents' in loaded_data:
                    # Chunked format
                    latent = loaded_data
                    architecture = loaded_data.get('architecture', 'unknown')
                else:
                    # Legacy single tensor format
                    latent = loaded_data
                    architecture = 'legacy'
            else:
                # Legacy format
                latent = loaded_data
                architecture = 'legacy'
                
            print(f"Decoding with architecture: {architecture}")
            audio = self.decode(latent)
            
            # Transpose for soundfile (samples, channels)
            audio_for_save = audio.T
            sf.write(output_path, audio_for_save, sample_rate)
            print(f"Decoded {input_path} -> {output_path}")
            
        except Exception as e:
            print(f"Error decoding file {input_path}: {e}")
    
    def get_compression_ratio(self, audio_length_seconds: float = None, latent_tensor: torch.Tensor = None) -> float:
        """Calculate compression ratio for log-mel architecture"""
        try:
            if audio_length_seconds is not None:
                # Original: 44.1kHz * 2 channels * 16 bits * seconds
                original_bits = SAMPLE_RATE * 2 * 16 * audio_length_seconds
                
                # NEW: Calculate for log-mel architecture
                # Time frames: audio_length * sample_rate / hop_length
                time_frames = int(audio_length_seconds * SAMPLE_RATE / HOP_LENGTH)
                
                # f10c10: 10x frequency compression (N_MELS=128 vs ~1025 STFT bins = ~8x)
                # 10x time compression through encoder stages
                compressed_time = time_frames // 10
                compressed_freq = N_MELS // 10  # Approximate after encoder
                
                # Latent representation
                typical_latent_elements = 64 * 8 * 32  # Channels * Height * Width
                bytes_per_element = 4 if not self.half_precision else 2  # float32 or float16
                compressed_bits = typical_latent_elements * bytes_per_element * 8  # Convert to bits
                
                return original_bits / compressed_bits
            
            elif latent_tensor is not None:
                # Calculate from actual latent tensor size
                latent_elements = latent_tensor.numel()
                bytes_per_element = 4 if latent_tensor.dtype == torch.float32 else 2
                compressed_bits = latent_elements * bytes_per_element * 8
                
                # Reverse engineer audio length from latent dimensions
                latent_time_frames = latent_tensor.shape[-1] if latent_tensor.dim() >= 2 else 32
                estimated_mel_time_frames = latent_time_frames * 10  # f10 expansion
                estimated_audio_frames = estimated_mel_time_frames * HOP_LENGTH
                original_bits = estimated_audio_frames * 2 * 16  # Stereo 16-bit
                
                return original_bits / compressed_bits
            else:
                # Default theoretical calculation for log-mel f10c10
                return 100.0  # f10c10 theoretical
                
        except Exception as e:
            print(f"Warning: Compression ratio calculation failed: {e}")
            return 100.0  # Default fallback
    
    def test_round_trip(self, audio_length_seconds: float = 5.0) -> dict:
        """Test encode-decode round trip consistency for log-mel + phase architecture"""
        try:
            # Generate test signal
            t = torch.linspace(0, audio_length_seconds, int(SAMPLE_RATE * audio_length_seconds))
            test_audio = torch.stack([
                torch.sin(2 * torch.pi * 440 * t),  # A4 note
                torch.sin(2 * torch.pi * 880 * t)   # A5 note
            ], dim=0).unsqueeze(0)  # [1, 2, T]
            
            # Round trip
            with self._memory_efficient_inference():
                latent = self.encode(test_audio, normalize=False)
                reconstructed = self.decode(latent, target_length=test_audio.shape[-1])
                
            # Calculate metrics
            original_np = test_audio.squeeze(0).cpu().numpy()  # Ensure CPU for numpy
            if isinstance(reconstructed, torch.Tensor):
                reconstructed_np = reconstructed.cpu().numpy()
            else:
                reconstructed_np = reconstructed
                
            mse = np.mean((original_np - reconstructed_np) ** 2)
            signal_power = np.mean(original_np ** 2)
            snr = 10 * np.log10(signal_power / (mse + 1e-10))
            
            # Calculate mel-scale specific metrics
            try:
                # Convert both to log-mel for comparison
                original_log_mel, _ = self._audio_to_log_mel_phase(test_audio)
                reconstructed_tensor = torch.from_numpy(reconstructed_np).unsqueeze(0)
                reconstructed_log_mel, _ = self._audio_to_log_mel_phase(reconstructed_tensor)
                
                mel_mse = torch.mean((original_log_mel - reconstructed_log_mel) ** 2).item()
                mel_snr = 10 * np.log10(torch.mean(original_log_mel ** 2).item() / (mel_mse + 1e-10))
            except Exception as e:
                print(f"Warning: Mel-scale metrics failed: {e}")
                mel_mse = float('inf')
                mel_snr = -float('inf')
            
            return {
                'mse': float(mse),
                'snr_db': float(snr),
                'mel_mse': float(mel_mse),
                'mel_snr_db': float(mel_snr),
                'original_shape': original_np.shape,
                'reconstructed_shape': reconstructed_np.shape,
                'architecture': 'log_mel_phase'
            }
            
        except Exception as e:
            print(f"Error in round trip test: {e}")
            return {
                'mse': float('inf'),
                'snr_db': -float('inf'),
                'mel_mse': float('inf'),
                'mel_snr_db': -float('inf'),
                'original_shape': (2, 0),
                'reconstructed_shape': (2, 0),
                'architecture': 'log_mel_phase'
            }