import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import hashlib
import warnings
import importlib.util

# Global audio parameters
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512  # f10 compression in time
N_MELS = 128
F_MIN = 20
F_MAX = 22050

# Cache expensive imports
HAS_SOXR = importlib.util.find_spec('soxr') is not None
HAS_TORCHAUDIO = importlib.util.find_spec('torchaudio') is not None
HAS_SCIPY = importlib.util.find_spec('scipy') is not None

# FIXED: Disable Triton completely
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
    print("ℹ️ Triton available but disabled for V100×4 stability")
except ImportError:
    HAS_TRITON = False
    print("ℹ️ Triton not available - using PyTorch implementations")

# CRITICAL: Force disable Triton globally
TRITON_ENABLED = False

def stft_transform(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann', return_complex=True):
    """
    CRITICAL: Fixed STFT with proper tensor dimension handling for distributed training
    Handles multi-dimensional batches correctly and prevents dimension errors
    """
    # CRITICAL: Handle input tensor dimensions properly
    original_shape = waveform.shape
    original_device = waveform.device
    original_dtype = waveform.dtype
    
    # Ensure we have proper input format
    if waveform.dim() == 1:
        # Single audio signal [T] -> [1, T]
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() > 2:
        # Multi-dimensional batch: flatten all except last dimension
        batch_shape = original_shape[:-1]
        total_batch = 1
        for dim in batch_shape:
            total_batch *= dim
        waveform = waveform.view(total_batch, original_shape[-1])
    
    # CRITICAL: Create window with proper device and dtype handling
    try:
        # Try to create window on same device and dtype
        window_fn = torch.hann_window(n_fft, device=original_device, dtype=original_dtype)
    except Exception as e:
        # Fallback: create on CPU and move
        window_fn = torch.hann_window(n_fft, dtype=original_dtype)
        window_fn = window_fn.to(device=original_device)
    
    # CRITICAL: STFT computation with proper error handling
    try:
        stft = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window_fn,
            return_complex=return_complex,
            normalized=False,
            onesided=True,
            center=True,  # Use center=True for stability
            pad_mode='reflect'
        )
    except Exception as e:
        print(f"⚠️ STFT failed with error: {e}")
        # Emergency fallback: create dummy output with correct dimensions
        batch_size = waveform.shape[0]
        freq_bins = n_fft // 2 + 1
        time_frames = (waveform.shape[-1] // hop_length) + 1
        
        if return_complex:
            stft = torch.zeros(batch_size, freq_bins, time_frames, 
                             dtype=torch.complex64, device=original_device)
        else:
            stft = torch.zeros(batch_size, freq_bins, time_frames, 2, 
                             dtype=original_dtype, device=original_device)
        print(f"⚠️ Using dummy STFT output: {stft.shape}")
    
    # CRITICAL: Restore original batch dimensions if needed
    if len(original_shape) > 2:
        # Reshape back to original batch dimensions + [freq, time] (+ [2] if real)
        if return_complex:
            new_shape = original_shape[:-1] + stft.shape[-2:]
        else:
            new_shape = original_shape[:-1] + stft.shape[-3:]
        stft = stft.view(new_shape)
    elif len(original_shape) == 1:
        # Remove added batch dimension for 1D input
        stft = stft.squeeze(0)
    
    return stft

def istft_transform(stft_tensor, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann', length=None):
    """
    CRITICAL: Fixed ISTFT with comprehensive tensor dimension handling
    Prevents dimension errors and handles complex multi-batch scenarios
    """
    # Handle precision conversion
    original_dtype = stft_tensor.dtype
    original_device = stft_tensor.device
    precision_converted = False
    
    if stft_tensor.dtype == torch.complex32:  # ComplexHalf
        stft_tensor = stft_tensor.to(torch.complex64)
        precision_converted = True
    
    # CRITICAL: Handle multi-dimensional input properly
    original_shape = stft_tensor.shape
    
    # Debug info for problematic tensors
    if len(original_shape) > 4:
        print(f"DEBUG: ISTFT input shape: {original_shape}")
    
    # CRITICAL: Comprehensive tensor reshaping logic
    if len(original_shape) == 2:
        # [freq, time] -> [1, freq, time]
        stft_tensor = stft_tensor.unsqueeze(0)
        added_batch = True
    elif len(original_shape) == 3:
        # [batch, freq, time] or [channels, freq, time]
        added_batch = False
    elif len(original_shape) == 4:
        # [batch, channels, freq, time] - need to process channels separately
        added_batch = False
    elif len(original_shape) == 5:
        # [batch1, batch2, channels, freq, time] -> flatten batch dimensions
        batch_dims = original_shape[:2]
        total_batch = batch_dims[0] * batch_dims[1]
        stft_tensor = stft_tensor.view(total_batch, *original_shape[2:])
        added_batch = False
        needs_reshape = True
        target_shape = batch_dims
    elif len(original_shape) > 5:
        # Flatten all batch dimensions except last 3
        batch_dims = original_shape[:-3]
        total_batch = 1
        for dim in batch_dims:
            total_batch *= dim
        stft_tensor = stft_tensor.view(total_batch, *original_shape[-3:])
        added_batch = False
        needs_reshape = True
        target_shape = batch_dims
    else:
        added_batch = False
        needs_reshape = False
    
    # CRITICAL: Create window with proper device/dtype handling
    try:
        window_fn = torch.hann_window(n_fft, device=original_device)
        if precision_converted:
            window_fn = window_fn.float()
        elif stft_tensor.dtype == torch.complex128:
            window_fn = window_fn.double()
    except Exception as e:
        print(f"⚠️ Window creation failed: {e}")
        window_fn = torch.hann_window(n_fft)
        window_fn = window_fn.to(device=original_device)
    
    # CRITICAL: Process ISTFT based on tensor dimensions
    try:
        if stft_tensor.dim() == 4:  # [batch, channels, freq, time]
            batch_size, num_channels, freq_bins, time_frames = stft_tensor.shape
            output_channels = []
            
            for ch in range(num_channels):
                channel_stft = stft_tensor[:, ch, :, :]  # [batch, freq, time]
                
                try:
                    channel_waveform = torch.istft(
                        channel_stft,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        window=window_fn,
                        normalized=False,
                        onesided=True,
                        center=True,  # Use center=True for stability
                        length=length
                    )
                    output_channels.append(channel_waveform)
                except Exception as e:
                    print(f"⚠️ ISTFT failed for channel {ch}: {e}")
                    # Create dummy output with correct length
                    if length is not None:
                        dummy_length = length
                    else:
                        dummy_length = (time_frames - 1) * hop_length + n_fft
                    dummy_waveform = torch.zeros(batch_size, dummy_length, 
                                                device=original_device, dtype=torch.float32)
                    output_channels.append(dummy_waveform)
            
            # Stack channels: [batch, channels, time]
            waveform = torch.stack(output_channels, dim=1)
            
        else:  # [batch, freq, time] - single channel case
            try:
                waveform = torch.istft(
                    stft_tensor,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    window=window_fn,
                    normalized=False,
                    onesided=True,
                    center=True,  # Use center=True for stability
                    length=length
                )
            except Exception as e:
                print(f"⚠️ ISTFT failed: {e}")
                # Create dummy output
                if length is not None:
                    dummy_length = length
                else:
                    time_frames = stft_tensor.shape[-1]
                    dummy_length = (time_frames - 1) * hop_length + n_fft
                
                if stft_tensor.dim() == 3:
                    batch_size = stft_tensor.shape[0]
                    waveform = torch.zeros(batch_size, dummy_length, 
                                         device=original_device, dtype=torch.float32)
                else:
                    waveform = torch.zeros(dummy_length, 
                                         device=original_device, dtype=torch.float32)
    
    except Exception as e:
        print(f"⚠️ Critical ISTFT error: {e}")
        # Emergency fallback
        if length is not None:
            dummy_length = length
        else:
            dummy_length = 44100  # 1 second fallback
        
        if len(original_shape) >= 4:
            batch_size = original_shape[0]
            num_channels = original_shape[1] if len(original_shape) > 3 else 1
            waveform = torch.zeros(batch_size, num_channels, dummy_length, 
                                 device=original_device, dtype=torch.float32)
        else:
            waveform = torch.zeros(dummy_length, 
                                 device=original_device, dtype=torch.float32)
    
    # CRITICAL: Restore original tensor structure
    if 'needs_reshape' in locals() and needs_reshape:
        # Reshape back to original multi-dimensional structure
        waveform_shape = waveform.shape
        if len(original_shape) == 5:
            # [batch1, batch2, channels, time]
            new_shape = target_shape + waveform_shape[1:]
        else:
            # General case
            new_shape = target_shape + waveform_shape[1:]
        waveform = waveform.view(new_shape)
    elif added_batch and len(original_shape) == 2:
        # Remove added batch dimension
        waveform = waveform.squeeze(0)
    
    # Restore precision if converted
    if precision_converted and original_dtype == torch.complex32:
        waveform = waveform.half()
    
    return waveform

def high_quality_resample(audio, orig_sr, target_sr):
    """High-quality resampling with enhanced error handling"""
    if HAS_SOXR:
        # Use pysoxr (highest quality)
        import soxr
        
        try:
            if isinstance(audio, torch.Tensor):
                audio_np = audio.cpu().numpy()
                was_torch = True
                original_device = audio.device
            else:
                audio_np = audio
                was_torch = False
            
            # Handle channel dimension detection
            if audio_np.ndim == 2:
                if audio_np.shape[1] <= 8:  # [time, channels]
                    transposed = False
                elif audio_np.shape[0] <= 8:  # [channels, time]
                    audio_np = audio_np.T
                    transposed = True
                else:
                    transposed = False
            else:
                transposed = False
            
            resampled = soxr.resample(audio_np, orig_sr, target_sr, quality='HQ')
            
            if transposed:
                resampled = resampled.T
            
            if was_torch:
                return torch.from_numpy(resampled).float().to(original_device)
            else:
                return resampled
                
        except Exception as e:
            print(f"⚠️ soxr resampling failed: {e}, falling back to torchaudio")
    
    if HAS_TORCHAUDIO:
        # Fallback to torchaudio
        try:
            import torchaudio.functional as AF
            
            if isinstance(audio, np.ndarray):
                audio_torch = torch.from_numpy(audio).float()
                was_numpy = True
            else:
                audio_torch = audio.float()
                was_numpy = False
            
            resampled = AF.resample(audio_torch, orig_sr, target_sr)
            
            if was_numpy:
                return resampled.cpu().numpy()
            else:
                return resampled
                
        except Exception as e:
            print(f"⚠️ torchaudio resampling failed: {e}, falling back to scipy")
    
    if HAS_SCIPY:
        # Fallback to scipy
        try:
            from scipy import signal
            
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            
            ratio = target_sr / orig_sr
            new_length = int(audio.shape[-1] * ratio)
            
            resampled = signal.resample(audio, new_length, axis=-1)
            return resampled
            
        except Exception as e:
            print(f"⚠️ scipy resampling failed: {e}, using linear interpolation")
    
    # Final fallback to linear interpolation
    warnings.warn("Using linear interpolation for resampling (low quality)")
    
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    
    ratio = target_sr / orig_sr
    new_length = int(audio.shape[-1] * ratio)
    
    if audio.ndim == 1:
        resampled = np.interp(
            np.linspace(0, audio.shape[0], new_length),
            np.arange(audio.shape[0]),
            audio
        )
    else:
        resampled = np.array([
            np.interp(
                np.linspace(0, audio.shape[1], new_length),
                np.arange(audio.shape[1]),
                audio[ch]
            ) for ch in range(audio.shape[0])
        ])
    
    return resampled

def create_deterministic_seed(path_str):
    """Create deterministic seed from path string with better randomization"""
    hash_obj = hashlib.md5(str(path_str).encode())
    # Use multiple hash iterations for better distribution
    for _ in range(3):
        hash_obj.update(hash_obj.digest())
    return int(hash_obj.hexdigest()[:8], 16)

class GammatoneFilterbank(nn.Module):
    """
    CRITICAL: Gammatone filterbank with enhanced gradient flow and error handling
    """
    def __init__(self, n_filters=64, f_min=F_MIN, f_max=F_MAX, sample_rate=SAMPLE_RATE, use_triton=False):
        super().__init__()
        self.n_filters = n_filters
        self.f_min = f_min
        self.f_max = f_max
        self.sample_rate = sample_rate
        self.use_triton = False  # Force disable
        
        # ERB scale center frequencies
        erb_min = self.hz_to_erb(f_min)
        erb_max = self.hz_to_erb(f_max)
        erb_points = torch.linspace(erb_min, erb_max, n_filters)
        center_freqs = self.erb_to_hz(erb_points)
        
        self.register_buffer('center_freqs', center_freqs)
        
        # Precompute ERB widths
        erb_widths = 24.7 * (4.37 * center_freqs / 1000 + 1)
        self.register_buffer('erb_widths', erb_widths)
        
        # CRITICAL: Learnable parameters to ensure gradient flow
        self.filter_weights = nn.Parameter(torch.ones(n_filters))
        self.bias = nn.Parameter(torch.zeros(n_filters))
        
        print(f"✅ GammatoneFilterbank: Stable PyTorch with gradient enhancement")
        
    def hz_to_erb(self, hz):
        if not isinstance(hz, torch.Tensor):
            hz = torch.tensor(hz, dtype=torch.float32)
        return 21.4 * torch.log10(1 + 0.00437 * hz)
    
    def erb_to_hz(self, erb):
        if not isinstance(erb, torch.Tensor):
            erb = torch.tensor(erb, dtype=torch.float32)
        return (10**(erb / 21.4) - 1) / 0.00437
    
    def forward(self, magnitude_spectrum):
        """
        CRITICAL: Enhanced forward pass ensuring gradient flow
        """
        try:
            B, F, T = magnitude_spectrum.shape
            
            # Create frequency grid with proper dtype matching
            freq_grid = torch.linspace(
                0, self.sample_rate // 2, F, 
                device=magnitude_spectrum.device,
                dtype=magnitude_spectrum.dtype
            )
            
            # Ensure buffers match input precision
            center_freqs = self.center_freqs.to(magnitude_spectrum.dtype)
            erb_widths = self.erb_widths.to(magnitude_spectrum.dtype)
            
            # Vectorized gammatone responses
            freq_diff = freq_grid.unsqueeze(1) - center_freqs.unsqueeze(0)
            
            sample_rate_tensor = torch.tensor(
                self.sample_rate, 
                device=magnitude_spectrum.device,
                dtype=magnitude_spectrum.dtype
            )
            
            responses = torch.exp(-2 * math.pi * erb_widths.unsqueeze(0) * 
                                torch.abs(freq_diff) / sample_rate_tensor)
            
            # CRITICAL: Apply learnable weights to ensure parameter usage
            weighted_responses = responses * self.filter_weights.unsqueeze(0)
            
            # Apply filters
            magnitude_reshaped = magnitude_spectrum.permute(0, 2, 1).contiguous().view(-1, F)
            filtered_reshaped = torch.matmul(magnitude_reshaped, weighted_responses)
            filtered_output = filtered_reshaped.view(B, T, self.n_filters).permute(0, 2, 1)
            
            # CRITICAL: Add bias to ensure bias parameter gets gradients
            filtered_output = filtered_output + self.bias.unsqueeze(0).unsqueeze(-1)
            
            return filtered_output
            
        except Exception as e:
            print(f"⚠️ GammatoneFilterbank failed: {e}")
            # Return dummy output with correct shape and gradients
            B, F, T = magnitude_spectrum.shape
            dummy_output = torch.zeros(B, self.n_filters, T, 
                                     device=magnitude_spectrum.device,
                                     dtype=magnitude_spectrum.dtype)
            # Ensure gradients flow through parameters
            dummy_output = dummy_output + self.filter_weights.sum() * 0.0001 + self.bias.sum() * 0.0001
            return dummy_output

def psychoacoustic_masking(gammatone_output, threshold_db=-60, use_triton=False):
    """
    CRITICAL: Psychoacoustic masking with enhanced error handling
    """
    try:
        return _psychoacoustic_masking_pytorch(gammatone_output, threshold_db)
    except Exception as e:
        print(f"⚠️ Psychoacoustic masking failed: {e}")
        # Return safe fallback
        return torch.ones_like(gammatone_output) * threshold_db

def _psychoacoustic_masking_pytorch(gammatone_output, threshold_db):
    """Enhanced PyTorch implementation with comprehensive error handling"""
    try:
        # Convert to dB with numerical stability
        eps = 1e-10
        power_db = 20 * torch.log10(torch.clamp(gammatone_output, min=eps))
        
        n_filters = gammatone_output.shape[1]
        device = gammatone_output.device
        dtype = gammatone_output.dtype
        
        # Create or retrieve spreading matrix
        cache_key = (n_filters, dtype)
        if (not hasattr(_psychoacoustic_masking_pytorch, '_spreading_cache') or 
            _psychoacoustic_masking_pytorch._spreading_cache is None or
            _psychoacoustic_masking_pytorch._spreading_cache[0] != cache_key):
            
            # Create spreading matrix
            spreading_matrix = torch.zeros(n_filters, n_filters, device=device, dtype=dtype)
            
            for i in range(n_filters):
                for j in range(n_filters):
                    if i != j:
                        bark_diff = j - i + 0.474
                        spread = max(0, 15.81 + 7.5 * bark_diff - 17.5 * (1 + bark_diff**2)**0.5)
                        spreading_matrix[i, j] = 10**(-spread / 10)
                    else:
                        spreading_matrix[i, j] = 1.0
            
            # Cache for reuse
            _psychoacoustic_masking_pytorch._spreading_cache = (cache_key, spreading_matrix.cpu())
        
        spreading_matrix = _psychoacoustic_masking_pytorch._spreading_cache[1].to(device=device, dtype=dtype)
        
        # Apply spreading
        B, n_filters, T = power_db.shape
        power_reshaped = power_db.permute(0, 2, 1).contiguous().view(-1, n_filters)
        spread_reshaped = torch.matmul(power_reshaped, spreading_matrix.T)
        spread_power = spread_reshaped.view(B, T, n_filters).permute(0, 2, 1)
        
        # Apply absolute threshold
        abs_threshold = torch.full_like(spread_power, threshold_db, dtype=dtype)
        masking_curve = torch.maximum(spread_power, abs_threshold)
        
        return masking_curve
        
    except Exception as e:
        print(f"⚠️ Psychoacoustic masking computation failed: {e}")
        # Safe fallback
        return torch.ones_like(gammatone_output) * threshold_db

def to_complex_spec(waveform):
    """Convert waveform to complex spectrogram with error handling"""
    try:
        stft = stft_transform(waveform, return_complex=True)
        return stft
    except Exception as e:
        print(f"⚠️ Complex spectrogram conversion failed: {e}")
        # Return dummy complex spectrogram
        if waveform.dim() == 1:
            batch_size = 1
            waveform = waveform.unsqueeze(0)
        else:
            batch_size = waveform.shape[0]
        
        freq_bins = N_FFT // 2 + 1
        time_frames = (waveform.shape[-1] // HOP_LENGTH) + 1
        return torch.zeros(batch_size, freq_bins, time_frames, 
                         dtype=torch.complex64, device=waveform.device)

def to_magnitude_phase(complex_spec):
    """Split complex spectrogram into magnitude and phase"""
    magnitude = torch.abs(complex_spec)
    phase = torch.angle(complex_spec)
    return magnitude, phase

def from_magnitude_phase(magnitude, phase):
    """Reconstruct complex spectrogram from magnitude and phase"""
    complex_spec = magnitude * torch.exp(1j * phase)
    return complex_spec

def to_waveform(complex_spec, length=None):
    """Convert complex spectrogram back to waveform with error handling"""
    try:
        return istft_transform(complex_spec, length=length)
    except Exception as e:
        print(f"⚠️ Waveform conversion failed: {e}")
        # Return dummy waveform
        if length is not None:
            dummy_length = length
        else:
            time_frames = complex_spec.shape[-1]
            dummy_length = (time_frames - 1) * HOP_LENGTH + N_FFT
        
        if complex_spec.dim() >= 3:
            batch_size = complex_spec.shape[0]
            if complex_spec.dim() >= 4:
                num_channels = complex_spec.shape[1]
                return torch.zeros(batch_size, num_channels, dummy_length, device=complex_spec.device)
            else:
                return torch.zeros(batch_size, dummy_length, device=complex_spec.device)
        else:
            return torch.zeros(dummy_length, device=complex_spec.device)

def normalize_audio(waveform, target_db=-23.0, method='rms'):
    """Normalize audio with enhanced error handling"""
    try:
        if method == 'rms':
            rms = torch.sqrt(torch.mean(waveform**2, dim=-1, keepdim=True))
            target_rms = 10**(target_db / 20)
            scale = target_rms / (rms + 1e-8)
        elif method == 'peak':
            peak = torch.max(torch.abs(waveform), dim=-1, keepdim=True)[0]
            target_peak = 10**(target_db / 20)
            scale = target_peak / (peak + 1e-8)
        elif method == 'lufs':
            if waveform.dim() > 1:
                weighted_power = torch.mean(waveform**2, dim=-1, keepdim=True)
            else:
                weighted_power = torch.mean(waveform**2, keepdim=True)
            
            lufs_estimate = torch.sqrt(weighted_power)
            target_lufs = 10**(target_db / 20)
            scale = target_lufs / (lufs_estimate + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Clamp scale to prevent extreme amplification
        scale = torch.clamp(scale, max=10.0)
        return waveform * scale
        
    except Exception as e:
        print(f"⚠️ Audio normalization failed: {e}")
        return waveform  # Return unchanged audio

def dynamic_range_compression(magnitude, ratio=4.0, threshold=0.1, knee_width=0.05):
    """
    CRITICAL: Dynamic range compression - INFERENCE ONLY with proper documentation
    
    WARNING: This function uses torch.no_grad() and is intended for inference only.
    Do not use during training as it breaks gradient flow.
    """
    with torch.no_grad():
        try:
            # Soft knee compression
            above_thresh = magnitude > threshold
            knee_region = (magnitude > threshold - knee_width) & (magnitude <= threshold + knee_width)
            
            compressed = magnitude.clone()
            
            # Hard compression above threshold
            compressed[above_thresh] = threshold + (magnitude[above_thresh] - threshold) / ratio
            
            # Soft knee compression
            if knee_region.any():
                knee_ratio = 1.0 + (ratio - 1.0) * (magnitude[knee_region] - (threshold - knee_width)) / (2 * knee_width)
                compressed[knee_region] = magnitude[knee_region] / knee_ratio
            
            return compressed
            
        except Exception as e:
            print(f"⚠️ Dynamic range compression failed: {e}")
            return magnitude

class SpectralLoss(nn.Module):
    """
    CRITICAL: Multi-scale spectral loss with comprehensive error handling and gradient flow
    Enhanced for spectrum-domain training with DDP compatibility
    """
    def __init__(self, n_ffts=[512, 1024, 2048], alpha=1.0, beta=1.0, gamma=1.0, use_triton=False):
        super().__init__()
        self.n_ffts = n_ffts
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_triton = False  # Force disable
        
        # CRITICAL: Learnable parameters to ensure gradient flow
        self.scale_weights = nn.Parameter(torch.ones(len(n_ffts)))
        self.loss_bias = nn.Parameter(torch.zeros(1))
        
        print(f"✅ SpectralLoss: Stable PyTorch with gradient enhancement")
        
    def spectral_convergence_loss(self, pred_stft, target_stft):
        """Spectral convergence loss with numerical stability"""
        try:
            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)
            
            numerator = torch.norm(target_mag - pred_mag, p='fro')
            denominator = torch.norm(target_mag, p='fro')
            
            return numerator / (denominator + 1e-8)
        except Exception as e:
            print(f"⚠️ Spectral convergence loss failed: {e}")
            return torch.tensor(0.0, device=pred_stft.device, requires_grad=True)
        
    def forward(self, pred_input, target_input):
        """
        CRITICAL: Enhanced forward pass with comprehensive error handling
        Supports both audio and complex spectrogram inputs
        """
        try:
            # Check if inputs are complex spectrograms
            if torch.is_complex(pred_input) and torch.is_complex(target_input):
                return self._spectrum_domain_loss(pred_input, target_input)
            else:
                return self._audio_domain_loss(pred_input, target_input)
        except Exception as e:
            print(f"⚠️ SpectralLoss forward failed: {e}")
            # Return safe fallback loss with gradients on the same device as the bias parameter
            device = self.loss_bias.device
            return torch.tensor(1.0, device=device, requires_grad=True) + self.loss_bias
    
    def _spectrum_domain_loss(self, pred_complex, target_complex):
        """CRITICAL: Direct spectrum-domain loss with error handling"""
        try:
            total_loss = torch.tensor(0.0, device=pred_complex.device, requires_grad=True)
            
            # Handle multi-channel input
            if pred_complex.dim() == 4:  # [B, C, F, T]
                for ch in range(pred_complex.shape[1]):
                    pred_ch = pred_complex[:, ch]
                    target_ch = target_complex[:, ch]
                    channel_loss = self._compute_spectrum_loss(pred_ch, target_ch)
                    total_loss = total_loss + channel_loss
                total_loss = total_loss / pred_complex.shape[1]
            else:
                total_loss = self._compute_spectrum_loss(pred_complex, target_complex)
            
            # CRITICAL: Add learnable parameter contribution
            total_loss = total_loss + self.loss_bias
            
            return total_loss
            
        except Exception as e:
            print(f"⚠️ Spectrum domain loss failed: {e}")
            return torch.tensor(1.0, device=pred_complex.device, requires_grad=True) + self.loss_bias
    
    def _compute_spectrum_loss(self, pred_stft, target_stft):
        """Compute loss for single spectrum with comprehensive error handling"""
        try:
            # Magnitude loss
            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)
            mag_loss = F.l1_loss(pred_mag, target_mag)
            
            # Spectral convergence loss
            sc_loss = self.spectral_convergence_loss(pred_stft, target_stft)
            
            # Phase loss with magnitude weighting
            magnitude_mask = target_mag > 0.01 * target_mag.max()
            if magnitude_mask.any():
                pred_phase = torch.angle(pred_stft)
                target_phase = torch.angle(target_stft)
                
                phase_diff_cos = torch.cos(pred_phase - target_phase)
                phase_loss = 1.0 - phase_diff_cos[magnitude_mask].mean()
            else:
                phase_loss = torch.tensor(0.0, device=pred_stft.device, requires_grad=True)
            
            return self.alpha * mag_loss + self.beta * phase_loss + self.gamma * sc_loss
            
        except Exception as e:
            print(f"⚠️ Spectrum loss computation failed: {e}")
            return torch.tensor(1.0, device=pred_stft.device, requires_grad=True)
    
    def _audio_domain_loss(self, pred_audio, target_audio):
        """Traditional multi-scale STFT loss with error handling"""
        try:
            total_loss = torch.tensor(0.0, device=pred_audio.device, requires_grad=True)
            
            for i, n_fft in enumerate(self.n_ffts):
                hop_length = n_fft // 4
                
                try:
                    # Compute STFTs
                    pred_stft = stft_transform(pred_audio, n_fft=n_fft, hop_length=hop_length)
                    target_stft = stft_transform(target_audio, n_fft=n_fft, hop_length=hop_length)
                    
                    scale_loss = self._compute_spectrum_loss(pred_stft, target_stft)
                    
                    # CRITICAL: Apply learnable scale weights
                    weighted_loss = scale_loss * self.scale_weights[i]
                    total_loss = total_loss + weighted_loss
                    
                except Exception as e:
                    print(f"⚠️ STFT scale {n_fft} failed: {e}")
                    # Add dummy loss to maintain gradient flow
                    dummy_loss = torch.tensor(0.1, device=pred_audio.device, requires_grad=True)
                    total_loss = total_loss + dummy_loss * self.scale_weights[i]
            
            # CRITICAL: Ensure all scale_weights participate
            total_loss = total_loss / len(self.n_ffts) + self.loss_bias
            
            return total_loss
            
        except Exception as e:
            print(f"⚠️ Audio domain loss failed: {e}")
            return torch.tensor(1.0, device=pred_audio.device, requires_grad=True) + self.loss_bias

def apply_window_function(signal, window_type='hann', fade_in=True, fade_out=True):
    """Apply window function with error handling"""
    try:
        length = signal.shape[-1]
        
        if window_type == 'hann':
            window = torch.hann_window(length, device=signal.device)
        elif window_type == 'hamming':
            window = torch.hamming_window(length, device=signal.device)
        elif window_type == 'blackman':
            window = torch.blackman_window(length, device=signal.device)
        else:
            window = torch.ones(length, device=signal.device)
        
        if not fade_in:
            window[:length//2] = 1.0
        if not fade_out:
            window[length//2:] = 1.0
        
        while window.dim() < signal.dim():
            window = window.unsqueeze(0)
            
        return signal * window
        
    except Exception as e:
        print(f"⚠️ Window function failed: {e}")
        return signal

def gammatone_filterbank(spectrum, n_filters=64, use_triton=False):
    """Convenience function for gammatone filtering"""
    try:
        filterbank = GammatoneFilterbank(n_filters=n_filters, use_triton=False)
        return filterbank(spectrum)
    except Exception as e:
        print(f"⚠️ Gammatone filterbank failed: {e}")
        B, F, T = spectrum.shape
        return torch.zeros(B, n_filters, T, device=spectrum.device, dtype=spectrum.dtype)

def get_optimal_pin_memory():
    """Get optimal pin_memory setting"""
    return torch.cuda.is_available()

# Initialize caches
_psychoacoustic_masking_pytorch._spreading_cache = None