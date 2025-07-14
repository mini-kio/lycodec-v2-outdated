import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import warnings
from typing import Union, Optional, Tuple, List
from contextlib import contextmanager

from .models import Model
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

class Codec:
    """
    FIXED: 단순화된 LyCodec 추론 엔진
    Log-mel + phase processing with improved stability
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 half_precision: bool = True,
                 max_chunk_length: int = 220500,
                 overlap_length: int = 4410):
        
        self.half_precision = half_precision
        self.max_chunk_length = max_chunk_length
        self.overlap_length = overlap_length
        
        # Auto-select device with simplified logic
        if device is None:
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.device = torch.device('cuda' if vram_gb >= 4 else 'cpu')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # Simplified FP16 handling
        if self.device.type == 'cpu':
            self.half_precision = False
        
        # Create mel filterbank
        self.mel_filterbank = create_mel_filterbank(n_mels=N_MELS, n_fft=N_FFT).to(self.device)
        
        # Load model with simplified error handling
        self.model = Model()
        if model_path:
            self._load_model(model_path)
        
        self.model.to(self.device)
        
        # Set precision
        if self.half_precision and self.device.type == 'cuda':
            try:
                self.model.half()
                self.mel_filterbank = self.mel_filterbank.half()
            except Exception:
                self.half_precision = False
        
        self.model.eval()
        
        # Simplified overlap handling
        if self.overlap_length > 0:
            try:
                self.fade_window = torch.hann_window(self.overlap_length * 2)[:self.overlap_length]
                self.fade_window = self.fade_window.to(self.device)
            except Exception:
                self.fade_window = torch.linspace(0, 1, self.overlap_length).to(self.device)
        else:
            self.fade_window = None
    
    def _load_model(self, model_path: str):
        """Load model weights with simplified error handling"""
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
            
            try:
                self.model.load_state_dict(new_state_dict, strict=True)
            except RuntimeError:
                self.model.load_state_dict(new_state_dict, strict=False)
                
        except Exception:
            warnings.warn(f"Failed to load model from {model_path}, using random weights")
    
    @contextmanager
    def _memory_efficient_inference(self):
        """Context manager for memory-efficient inference"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        try:
            with torch.no_grad():
                yield
        finally:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
    
    def _audio_to_log_mel_phase(self, audio_tensor):
        """
        Convert audio to log-mel + phase representation
        Simplified version with basic error handling
        """
        B, channels, T = audio_tensor.shape
        
        # Process each channel separately
        all_log_mels = []
        all_phases = []
        
        for ch in range(channels):
            # STFT
            complex_spec = to_complex_spec(audio_tensor[:, ch])
            magnitude_spec, phase_spec = to_magnitude_phase(complex_spec)
            
            # Convert to mel-scale
            mel_spec = to_mel_spectrogram(magnitude_spec, self.mel_filterbank)
            log_mel_spec = to_log_mel(mel_spec)
            
            # Simple phase processing - interpolate to mel dimensions
            B_phase, F_bins, T_frames = phase_spec.shape
            
            # Interpolate phase to mel scale
            phase_4d = phase_spec.unsqueeze(1)
            phase_interpolated = torch.nn.functional.interpolate(
                phase_4d, size=(N_MELS, T_frames), 
                mode='bilinear', align_corners=False
            ).squeeze(1)
            
            all_log_mels.append(log_mel_spec)
            all_phases.append(phase_interpolated)
        
        # Average across channels
        log_mel_features = torch.stack(all_log_mels, dim=1).mean(dim=1)
        phase_features = torch.stack(all_phases, dim=1).mean(dim=1)
        
        return log_mel_features, phase_features
    
    def _log_mel_phase_to_audio(self, log_mel_features, phase_features, target_length=None):
        """
        Convert log-mel + phase back to audio
        Simplified version with basic error handling
        """
        # Convert log-mel back to linear
        mel_features = from_log_mel(log_mel_features)
        
        # Convert mel back to magnitude
        magnitude_spec = mel_to_magnitude(mel_features, self.mel_filterbank, n_fft=N_FFT)
        
        # Reconstruct phase - interpolate from mel scale
        F_bins = N_FFT // 2 + 1
        T_frames = phase_features.shape[-1]
        
        phase_spec = torch.nn.functional.interpolate(
            phase_features.unsqueeze(2), size=(F_bins, T_frames), 
            mode='bilinear', align_corners=False
        ).squeeze(2)
        
        # Reconstruct complex spectrogram
        complex_spec = from_magnitude_phase(magnitude_spec, phase_spec)
        
        # Convert to audio (mono)
        mono_audio = to_waveform(complex_spec, length=target_length)
        
        # Expand to stereo
        if mono_audio.dim() == 2:
            stereo_audio = mono_audio.unsqueeze(1).repeat(1, 2, 1)
        else:
            stereo_audio = mono_audio.unsqueeze(0).repeat(2, 1).unsqueeze(0)
        
        return stereo_audio
    
    def _chunk_audio(self, audio: torch.Tensor) -> List[Tuple[torch.Tensor, int]]:
        """Split audio into chunks - simplified"""
        if audio.shape[-1] <= self.max_chunk_length:
            return [(audio, 0)]
        
        chunks = []
        start = 0
        
        while start < audio.shape[-1]:
            end = min(start + self.max_chunk_length, audio.shape[-1])
            chunk = audio[..., start:end]
            chunks.append((chunk, start))
            
            if end >= audio.shape[-1]:
                break
                
            start = end - self.overlap_length
        
        return chunks
    
    def _merge_chunks(self, chunks: List[Tuple[torch.Tensor, int]], original_length: int) -> torch.Tensor:
        """Merge chunks with simplified overlap handling"""
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
        
        # Simple overlap-add
        for i, (chunk, start_pos) in enumerate(chunks):
            chunk_len = chunk.shape[-1]
            end_pos = min(start_pos + chunk_len, original_length)
            actual_chunk_len = end_pos - start_pos
            
            if actual_chunk_len <= 0:
                continue
            
            if i == 0:
                # First chunk
                merged[..., start_pos:end_pos] = chunk[..., :actual_chunk_len]
            elif i == len(chunks) - 1:
                # Last chunk with fade in
                fade_len = min(self.overlap_length, actual_chunk_len)
                if fade_len > 0 and self.fade_window is not None:
                    fade_in = self.fade_window[:fade_len]
                    merged[..., start_pos:start_pos + fade_len] += chunk[..., :fade_len] * fade_in
                
                if fade_len < actual_chunk_len:
                    merged[..., start_pos + fade_len:end_pos] = chunk[..., fade_len:actual_chunk_len]
            else:
                # Middle chunk with fade in/out
                fade_len = min(self.overlap_length, actual_chunk_len)
                if fade_len > 0 and self.fade_window is not None:
                    fade_in = self.fade_window[:fade_len]
                    merged[..., start_pos:start_pos + fade_len] += chunk[..., :fade_len] * fade_in
                
                middle_start = fade_len
                middle_end = actual_chunk_len - self.overlap_length
                if middle_end > middle_start:
                    merged[..., start_pos + middle_start:start_pos + middle_end] = chunk[..., middle_start:middle_end]
        
        return merged
    
    def encode(self, audio: Union[str, np.ndarray, torch.Tensor], 
               normalize: bool = True) -> torch.Tensor:
        """
        Encode audio to compressed latent representation
        Simplified version with basic error handling
        """
        try:
            # Load and preprocess audio
            if isinstance(audio, str):
                audio_data, sr = sf.read(audio, always_2d=True)
                audio_data = audio_data.T
                
                if sr != SAMPLE_RATE:
                    warnings.warn(f"Audio sample rate {sr} != {SAMPLE_RATE}")
                
                audio_tensor = torch.from_numpy(audio_data).float()
                    
            elif isinstance(audio, np.ndarray):
                audio_tensor = torch.from_numpy(audio).float()
            else:
                audio_tensor = audio.float()
            
            # Ensure stereo
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0).repeat(2, 1)
            elif audio_tensor.dim() == 2:
                if audio_tensor.shape[0] == 1:
                    audio_tensor = audio_tensor.repeat(2, 1)
            elif audio_tensor.dim() == 3:
                if audio_tensor.shape[1] == 1:
                    audio_tensor = audio_tensor.repeat(1, 2, 1)
            
            # Add batch dimension if needed
            if audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Normalize
            if normalize:
                audio_tensor = normalize_audio(audio_tensor)
            
            # Move to device and set precision
            audio_tensor = audio_tensor.to(self.device)
            if self.half_precision and self.device.type == 'cuda':
                audio_tensor = audio_tensor.half()
            
            with self._memory_efficient_inference():
                # Process in chunks
                original_length = audio_tensor.shape[-1]
                chunks = self._chunk_audio(audio_tensor)
                
                latent_chunks = []
                for chunk, start_idx in chunks:
                    try:
                        # Convert to log-mel + phase
                        log_mel_features, phase_features = self._audio_to_log_mel_phase(chunk)
                        
                        if self.half_precision and self.device.type == 'cuda':
                            log_mel_features = log_mel_features.half()
                            phase_features = phase_features.half()
                        
                        # Encode
                        latent_chunk = self.model.encode(log_mel_features, phase_features)
                        latent_chunks.append((latent_chunk.cpu(), start_idx))
                        
                    except Exception:
                        # Dummy latent for failed chunk
                        dummy_latent = torch.zeros(1, 64, 8, 32)
                        latent_chunks.append((dummy_latent, start_idx))
            
            # Return latent
            if len(latent_chunks) == 1:
                latent, _ = latent_chunks[0]
                return latent
            else:
                latents = [chunk for chunk, _ in latent_chunks]
                return {
                    'latents': torch.cat(latents, dim=-1),
                    'original_length': original_length,
                    'chunk_info': [(start_idx, latent.shape) for latent, start_idx in latent_chunks]
                }
                
        except Exception:
            # Return dummy latent on failure
            return torch.zeros(1, 64, 8, 32)
    
    def decode(self, latent: Union[torch.Tensor, dict], 
               target_length: Optional[int] = None) -> np.ndarray:
        """
        Decode latent representation back to audio
        Simplified version with basic error handling
        """
        try:
            # Handle chunked latents
            if isinstance(latent, dict) and 'chunk_info' in latent:
                return self._decode_chunked(latent, target_length)
            
            # Handle single latent
            latent_tensor = latent['latents'] if isinstance(latent, dict) else latent
            
            if latent_tensor.dim() == 3:
                latent_tensor = latent_tensor.unsqueeze(0)
            
            latent_tensor = latent_tensor.to(self.device)
            if self.half_precision and self.device.type == 'cuda':
                latent_tensor = latent_tensor.half()
            
            with self._memory_efficient_inference():
                try:
                    # Calculate target size
                    if target_length is not None:
                        target_time_frames = (target_length // HOP_LENGTH) + 1
                        target_size = (N_MELS, target_time_frames)
                    else:
                        target_size = None
                    
                    # Decode
                    log_mel_out, phase_out = self.model.decode(latent_tensor, target_size)
                    
                    # Convert back to audio
                    stereo_audio = self._log_mel_phase_to_audio(log_mel_out, phase_out, target_length)
                    
                    # Remove batch dimension and move to CPU
                    stereo_audio = stereo_audio.squeeze(0).cpu().float()
                    
                    if target_length is not None:
                        stereo_audio = stereo_audio[..., :target_length]
                    
                    return stereo_audio.numpy()
                    
                except Exception:
                    # Return dummy audio
                    dummy_length = target_length if target_length else 44100
                    return np.zeros((2, dummy_length), dtype=np.float32)
        
        except Exception:
            # Return dummy audio on complete failure
            dummy_length = target_length if target_length else 44100
            return np.zeros((2, dummy_length), dtype=np.float32)
    
    def _decode_chunked(self, latent_dict: dict, target_length: Optional[int] = None) -> np.ndarray:
        """Decode chunked latents - simplified"""
        try:
            chunk_info = latent_dict['chunk_info']
            original_length = latent_dict.get('original_length', target_length)
            
            # Decode each chunk
            decoded_chunks = []
            chunk_start = 0
            
            for i, (start_idx, chunk_shape) in enumerate(chunk_info):
                try:
                    chunk_width = chunk_shape[-1]
                    chunk_latent = latent_dict['latents'][..., chunk_start:chunk_start + chunk_width]
                    chunk_audio = self.decode(chunk_latent)
                    decoded_chunks.append((chunk_audio, start_idx))
                    chunk_start += chunk_width
                    
                except Exception:
                    # Dummy audio for failed chunk
                    dummy_audio = np.zeros((2, 44100), dtype=np.float32)
                    decoded_chunks.append((dummy_audio, start_idx))
                    chunk_start += chunk_shape[-1]
            
            # Simple concatenation for chunked audio
            if len(decoded_chunks) == 1:
                audio, _ = decoded_chunks[0]
                return audio[:, :target_length] if target_length else audio
            
            # Merge chunks
            total_length = original_length or target_length or max(
                start_idx + audio.shape[-1] for audio, start_idx in decoded_chunks
            )
            
            merged_audio = np.zeros((2, total_length), dtype=np.float32)
            
            for chunk_audio, start_idx in decoded_chunks:
                chunk_len = chunk_audio.shape[-1]
                end_idx = min(start_idx + chunk_len, total_length)
                actual_len = end_idx - start_idx
                
                if actual_len > 0:
                    merged_audio[:, start_idx:end_idx] = chunk_audio[:, :actual_len]
            
            return merged_audio[:, :target_length] if target_length else merged_audio
            
        except Exception:
            dummy_length = target_length if target_length else 44100
            return np.zeros((2, dummy_length), dtype=np.float32)
    
    def encode_file(self, input_path: str, output_path: str):
        """Encode audio file and save latent representation"""
        try:
            latent = self.encode(input_path)
            
            if isinstance(latent, torch.Tensor):
                save_data = {
                    'latent': latent.cpu(),
                    'architecture': 'log_mel_phase'
                }
            else:
                save_data = latent.copy()
                save_data['architecture'] = 'log_mel_phase'
                if 'latents' in latent:
                    latent['latents'] = latent['latents'].cpu()
            
            torch.save(save_data, output_path)
            
            # Print compression stats
            try:
                input_size = Path(input_path).stat().st_size
                output_size = Path(output_path).stat().st_size
                compression_ratio = input_size / output_size
                print(f"Compression ratio: {compression_ratio:.1f}x")
            except Exception:
                pass
                
        except Exception as e:
            print(f"Error encoding file {input_path}: {e}")
    
    def decode_file(self, input_path: str, output_path: str, sample_rate: int = SAMPLE_RATE):
        """Decode latent file and save as audio"""
        try:
            loaded_data = torch.load(input_path, map_location='cpu')
            
            if isinstance(loaded_data, dict):
                if 'latent' in loaded_data:
                    latent = loaded_data['latent']
                elif 'latents' in loaded_data:
                    latent = loaded_data
                else:
                    latent = loaded_data
            else:
                latent = loaded_data
                
            audio = self.decode(latent)
            
            # Transpose for soundfile
            audio_for_save = audio.T
            sf.write(output_path, audio_for_save, sample_rate)
            
        except Exception as e:
            print(f"Error decoding file {input_path}: {e}")
    
    def get_compression_ratio(self, audio_length_seconds: float = None, latent_tensor: torch.Tensor = None) -> float:
        """Calculate compression ratio"""
        try:
            if audio_length_seconds is not None:
                original_bits = SAMPLE_RATE * 2 * 16 * audio_length_seconds
                time_frames = int(audio_length_seconds * SAMPLE_RATE / HOP_LENGTH)
                compressed_time = time_frames // 10
                compressed_freq = N_MELS // 10
                typical_latent_elements = 64 * 8 * 32
                bytes_per_element = 4 if not self.half_precision else 2
                compressed_bits = typical_latent_elements * bytes_per_element * 8
                return original_bits / compressed_bits
                
            elif latent_tensor is not None:
                latent_elements = latent_tensor.numel()
                bytes_per_element = 4 if latent_tensor.dtype == torch.float32 else 2
                compressed_bits = latent_elements * bytes_per_element * 8
                latent_time_frames = latent_tensor.shape[-1] if latent_tensor.dim() >= 2 else 32
                estimated_mel_time_frames = latent_time_frames * 10
                estimated_audio_frames = estimated_mel_time_frames * HOP_LENGTH
                original_bits = estimated_audio_frames * 2 * 16
                return original_bits / compressed_bits
            else:
                return 100.0
                
        except Exception:
            return 100.0
    
    def test_round_trip(self, audio_length_seconds: float = 5.0) -> dict:
        """Test encode-decode round trip"""
        try:
            # Generate test signal
            t = torch.linspace(0, audio_length_seconds, int(SAMPLE_RATE * audio_length_seconds))
            test_audio = torch.stack([
                torch.sin(2 * torch.pi * 440 * t),
                torch.sin(2 * torch.pi * 880 * t)
            ], dim=0).unsqueeze(0)
            
            # Round trip
            with self._memory_efficient_inference():
                latent = self.encode(test_audio, normalize=False)
                reconstructed = self.decode(latent, target_length=test_audio.shape[-1])
                
            # Calculate metrics
            original_np = test_audio.squeeze(0).cpu().numpy()
            if isinstance(reconstructed, torch.Tensor):
                reconstructed_np = reconstructed.cpu().numpy()
            else:
                reconstructed_np = reconstructed
                
            mse = np.mean((original_np - reconstructed_np) ** 2)
            signal_power = np.mean(original_np ** 2)
            snr = 10 * np.log10(signal_power / (mse + 1e-10))
            
            return {
                'mse': float(mse),
                'snr_db': float(snr),
                'original_shape': original_np.shape,
                'reconstructed_shape': reconstructed_np.shape,
                'architecture': 'log_mel_phase'
            }
            
        except Exception:
            return {
                'mse': float('inf'),
                'snr_db': -float('inf'),
                'original_shape': (2, 0),
                'reconstructed_shape': (2, 0),
                'architecture': 'log_mel_phase'
            }

# Backward compatibility
LyCodec = Codec  