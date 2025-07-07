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

# FIXED: Disable Triton completely due to compilation issues
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
    print("ℹ️ Triton available but disabled for V100×4 stability")
except ImportError:
    HAS_TRITON = False
    print("ℹ️ Triton not available - using PyTorch implementations")

# FIXED: Force disable Triton globally to prevent compilation errors
TRITON_ENABLED = False

def stft_transform(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann', return_complex=True):
    """
    STFT with phase preservation for high quality reconstruction - DISTRIBUTED TRAINING SAFE
    Simplified version to avoid hanging in multi-GPU training
    """
    # Handle multi-dimensional input - STFT expects 1D or 2D tensor
    original_shape = waveform.shape
    
    # Ensure we have a 2D tensor [batch, time] or [channels, time]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # Add batch dimension
    elif waveform.dim() > 2:
        # Flatten all dimensions except the last one (time)
        batch_dims = original_shape[:-1]
        total_batch = 1
        for dim in batch_dims:
            total_batch *= dim
        waveform = waveform.view(total_batch, original_shape[-1])
    
    # Use simpler window creation to avoid device/dtype issues
    try:
        # Try to create window on same device and dtype
        if waveform.device.type == 'cuda':
            window_fn = torch.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype)
        else:
            window_fn = torch.hann_window(n_fft, dtype=waveform.dtype)
            
        stft = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window_fn,
            return_complex=return_complex,
            normalized=False,
            onesided=True,
            center=True,
            pad_mode='reflect'
        )
    except Exception as e:
        # Fallback: create window separately and ensure compatibility
        window_fn = torch.hann_window(n_fft)
        window_fn = window_fn.to(device=waveform.device, dtype=waveform.dtype)
        
        stft = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window_fn,
            return_complex=return_complex,
            normalized=False,
            onesided=True,
            center=True,
            pad_mode='reflect'
        )
    
    # Restore original batch dimensions if needed
    if len(original_shape) > 2:
        # Reshape back to original batch dimensions + [freq, time]
        new_shape = original_shape[:-1] + stft.shape[-2:]
        stft = stft.view(new_shape)
    elif original_shape[0] == 1 and len(original_shape) == 1:
        # Remove added batch dimension for 1D input
        stft = stft.squeeze(0)
    
    return stft

def istft_transform(stft_tensor, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann', length=None):
    """
    Inverse STFT with phase preservation - STREAMING OPTIMIZED
    Handles multi-dimensional batches for distributed training
    """
    # Handle ComplexHalf compatibility
    original_dtype = stft_tensor.dtype
    if stft_tensor.dtype == torch.complex32:  # ComplexHalf
        stft_tensor = stft_tensor.to(torch.complex64)
        precision_converted = True
    else:
        precision_converted = False
    
    # Handle multi-dimensional batches (for distributed training)
    original_shape = stft_tensor.shape
    
    # Print debug info for problematic tensors
    if len(original_shape) > 4:
        print(f"DEBUG: ISTFT input shape: {original_shape}")
    
    # Reshape to proper format for ISTFT: [batch, freq, time] or [batch, channels, freq, time]
    if len(original_shape) == 5:
        # Shape: [batch1, batch2, channels, freq, time] -> [batch1*batch2, channels, freq, time]
        batch_size = original_shape[0] * original_shape[1]
        stft_tensor = stft_tensor.view(batch_size, *original_shape[2:])
    elif len(original_shape) > 5:
        # Flatten all batch dimensions except last 3 (channels, freq, time)
        batch_dims = original_shape[:-3]
        total_batch = 1
        for dim in batch_dims:
            total_batch *= dim
        stft_tensor = stft_tensor.view(total_batch, *original_shape[-3:])
    elif len(original_shape) == 3:
        # Add batch dimension: [channels, freq, time] -> [1, channels, freq, time]
        stft_tensor = stft_tensor.unsqueeze(0)
    
    # Now stft_tensor should be [batch, channels, freq, time]
    # ISTFT expects [batch, freq, time] for each channel, so we need to process channels separately
    if stft_tensor.dim() == 4:  # [batch, channels, freq, time]
        batch_size, num_channels, freq_bins, time_frames = stft_tensor.shape
        output_channels = []
        
        for ch in range(num_channels):
            channel_stft = stft_tensor[:, ch, :, :]  # [batch, freq, time]
            
            if stft_tensor.device.type == 'cuda':
                window_fn = torch.hann_window(n_fft, device=stft_tensor.device)
            else:
                window_fn = torch.hann_window(n_fft)
            
            # Ensure window has same precision as input
            if precision_converted:
                window_fn = window_fn.float()
            elif stft_tensor.dtype == torch.complex128:
                window_fn = window_fn.double()
            
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
            except RuntimeError as e:
                raise e
            
            output_channels.append(channel_waveform)
        
        # Stack channels: [batch, channels, time]
        waveform = torch.stack(output_channels, dim=1)
        
    else:  # [batch, freq, time] - single channel case
        if stft_tensor.device.type == 'cuda':
            window_fn = torch.hann_window(n_fft, device=stft_tensor.device)
        else:
            window_fn = torch.hann_window(n_fft)
        
        # Ensure window has same precision as input
        if precision_converted:
            window_fn = window_fn.float()
        elif stft_tensor.dtype == torch.complex128:
            window_fn = window_fn.double()
        
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
        except RuntimeError as e:
            raise e
    
    # Restore original batch shape if it was multi-dimensional
    if len(original_shape) == 5:
        # Reshape back to original 5D shape
        batch1, batch2 = original_shape[0], original_shape[1]
        waveform = waveform.view(batch1, batch2, *waveform.shape[1:])
    elif len(original_shape) > 5:
        # Reshape back to original multi-dimensional batch shape
        waveform_shape = waveform.shape  # [total_batch, channels, time]
        batch_dims = original_shape[:-3]  # Original batch dimensions
        new_shape = batch_dims + waveform_shape[1:]  # [batch_dims..., channels, time]
        waveform = waveform.view(new_shape)
    elif len(original_shape) == 3:
        # Remove added batch dimension if original didn't have it
        waveform = waveform.squeeze(0)
    
    if precision_converted and original_dtype == torch.complex32:
        waveform = waveform.half()
    
    return waveform

def high_quality_resample(audio, orig_sr, target_sr):
    """High-quality resampling with cached import checking"""
    if HAS_SOXR:
        # Use pysoxr (highest quality and speed)
        import soxr
        
        if isinstance(audio, torch.Tensor):
            audio_np = audio.numpy()
            was_torch = True
        else:
            audio_np = audio
            was_torch = False
            
        # Simplified channel detection logic
        if audio_np.ndim == 2 and audio_np.shape[1] <= 8:
            transposed = False
        elif audio_np.ndim == 2 and audio_np.shape[0] <= 8:
            audio_np = audio_np.T
            transposed = True
        else:
            transposed = False
            
        resampled = soxr.resample(audio_np, orig_sr, target_sr, quality='HQ')
        
        if transposed:
            resampled = resampled.T
            
        if was_torch:
            return torch.from_numpy(resampled).float()
        else:
            return resampled
            
    elif HAS_TORCHAUDIO:
        # Fallback to torchaudio
        import torchaudio.functional as AF
        if isinstance(audio, np.ndarray):
            audio_torch = torch.from_numpy(audio).float()
        else:
            audio_torch = audio.float()
            
        resampled = AF.resample(audio_torch, orig_sr, target_sr)
        
        if isinstance(audio, np.ndarray):
            return resampled.numpy()
        else:
            return resampled
            
    elif HAS_SCIPY:
        # Fallback to scipy
        warnings.warn("pysoxr and torchaudio not available, using scipy")
        
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
            
        ratio = target_sr / orig_sr
        new_length = int(audio.shape[-1] * ratio)
        
        from scipy import signal
        resampled = signal.resample(audio, new_length, axis=-1)
        return resampled
    else:
        # Final fallback to linear interpolation
        warnings.warn("No high-quality resampling available, using linear interpolation")
        
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
            
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
    """Create deterministic seed from path string"""
    return int(hashlib.md5(str(path_str).encode()).hexdigest()[:8], 16)

class GammatoneFilterbank(nn.Module):
    """
    Gammatone filterbank for psychoacoustic analysis - STABLE PYTORCH VERSION
    """
    def __init__(self, n_filters=64, f_min=F_MIN, f_max=F_MAX, sample_rate=SAMPLE_RATE, use_triton=False):
        super().__init__()
        self.n_filters = n_filters
        self.f_min = f_min
        self.f_max = f_max
        self.sample_rate = sample_rate
        # FIXED: Always use PyTorch implementation
        self.use_triton = False
        
        # ERB scale center frequencies
        erb_min = self.hz_to_erb(f_min)
        erb_max = self.hz_to_erb(f_max)
        erb_points = torch.linspace(erb_min, erb_max, n_filters)
        center_freqs = self.erb_to_hz(erb_points)
        
        self.register_buffer('center_freqs', center_freqs)
        
        # Precompute ERB widths for efficiency
        erb_widths = 24.7 * (4.37 * center_freqs / 1000 + 1)
        self.register_buffer('erb_widths', erb_widths)
        
        print(f"✅ GammatoneFilterbank: Using stable PyTorch implementation")
        
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
        Apply gammatone filterbank to magnitude spectrum
        Args:
            magnitude_spectrum: [B, F, T] - magnitude spectrum
        Returns:
            filtered: [B, n_filters, T] - gammatone filtered output
        """
        # Always use PyTorch implementation for stability
        return self._forward_pytorch(magnitude_spectrum)
    
    def _forward_pytorch(self, magnitude_spectrum):
        """Stable PyTorch implementation"""
        B, F, T = magnitude_spectrum.shape
        
        # Create frequency grid - match input dtype and device for FP16 compatibility
        freq_grid = torch.linspace(
            0, self.sample_rate // 2, F, 
            device=magnitude_spectrum.device,
            dtype=magnitude_spectrum.dtype
        )
        
        # Ensure center_freqs and erb_widths match input precision
        center_freqs = self.center_freqs.to(magnitude_spectrum.dtype)
        erb_widths = self.erb_widths.to(magnitude_spectrum.dtype)
        
        # Vectorized gammatone responses computation
        freq_diff = freq_grid.unsqueeze(1) - center_freqs.unsqueeze(0)
        
        # Gammatone filter responses
        sample_rate_tensor = torch.tensor(
            self.sample_rate, 
            device=magnitude_spectrum.device,
            dtype=magnitude_spectrum.dtype
        )
        responses = torch.exp(-2 * math.pi * erb_widths.unsqueeze(0) * 
                            torch.abs(freq_diff) / sample_rate_tensor)  # [F, n_filters]
        
        # Apply filters: [B, F, T] @ [F, n_filters] -> [B, n_filters, T]
        magnitude_reshaped = magnitude_spectrum.permute(0, 2, 1).contiguous().view(-1, F)  # [B*T, F]
        filtered_reshaped = torch.matmul(magnitude_reshaped, responses)  # [B*T, n_filters]
        filtered_output = filtered_reshaped.view(B, T, self.n_filters).permute(0, 2, 1)  # [B, n_filters, T]
        
        return filtered_output

def psychoacoustic_masking(gammatone_output, threshold_db=-60, use_triton=False):
    """
    Compute psychoacoustic masking curve - STABLE PYTORCH VERSION
    """
    # Always use PyTorch implementation
    return _psychoacoustic_masking_pytorch(gammatone_output, threshold_db)

def _psychoacoustic_masking_pytorch(gammatone_output, threshold_db):
    """Stable PyTorch implementation"""
    # Convert to dB with improved numerical stability
    eps = 1e-10
    power_db = 20 * torch.log10(torch.clamp(gammatone_output, min=eps))
    
    n_filters = gammatone_output.shape[1]
    device = gammatone_output.device
    dtype = gammatone_output.dtype
    
    # Precompute spreading matrix if not cached
    cache_key = (n_filters, dtype)
    if (not hasattr(_psychoacoustic_masking_pytorch, '_spreading_cache') or 
        _psychoacoustic_masking_pytorch._spreading_cache is None or
        _psychoacoustic_masking_pytorch._spreading_cache[0] != cache_key):
        
        # Create spreading matrix with correct dtype
        spreading_matrix = torch.zeros(n_filters, n_filters, device=device, dtype=dtype)
        
        for i in range(n_filters):
            for j in range(n_filters):
                if i != j:
                    # Simplified spreading function
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
    power_reshaped = power_db.permute(0, 2, 1).contiguous().view(-1, n_filters)  # [B*T, n_filters]
    spread_reshaped = torch.matmul(power_reshaped, spreading_matrix.T)  # [B*T, n_filters]
    spread_power = spread_reshaped.view(B, T, n_filters).permute(0, 2, 1)  # [B, n_filters, T]
    
    # Apply absolute threshold
    abs_threshold = torch.full_like(spread_power, threshold_db, dtype=dtype)
    masking_curve = torch.maximum(spread_power, abs_threshold)
    
    return masking_curve

def to_complex_spec(waveform):
    """Convert waveform to complex spectrogram representation"""
    stft = stft_transform(waveform, return_complex=True)
    return stft

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
    """Convert complex spectrogram back to waveform"""
    return istft_transform(complex_spec, length=length)

def normalize_audio(waveform, target_db=-23.0, method='rms'):
    """Normalize audio to target dB level"""
    if method == 'rms':
        # RMS normalization
        rms = torch.sqrt(torch.mean(waveform**2, dim=-1, keepdim=True))
        target_rms = 10**(target_db / 20)
        scale = target_rms / (rms + 1e-8)
    elif method == 'peak':
        # Peak normalization
        peak = torch.max(torch.abs(waveform), dim=-1, keepdim=True)[0]
        target_peak = 10**(target_db / 20)
        scale = target_peak / (peak + 1e-8)
    elif method == 'lufs':
        # Simplified LUFS-like normalization
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

def dynamic_range_compression(magnitude, ratio=4.0, threshold=0.1, knee_width=0.05):
    """
    Apply dynamic range compression to magnitude spectrum - INFERENCE ONLY
    
    WARNING: This function uses torch.no_grad() and is intended for inference only.
    Do not use during training as it breaks gradient flow.
    """
    with torch.no_grad():
        # Soft knee compression for smoother transitions
        above_thresh = magnitude > threshold
        
        # Soft knee region
        knee_region = (magnitude > threshold - knee_width) & (magnitude <= threshold + knee_width)
        
        compressed = magnitude.clone()
        
        # Hard compression above threshold
        compressed[above_thresh] = threshold + (magnitude[above_thresh] - threshold) / ratio
        
        # Soft knee compression
        if knee_region.any():
            knee_ratio = 1.0 + (ratio - 1.0) * (magnitude[knee_region] - (threshold - knee_width)) / (2 * knee_width)
            compressed[knee_region] = magnitude[knee_region] / knee_ratio
    
    return compressed

class SpectralLoss(nn.Module):
    """Multi-scale spectral loss for high-quality reconstruction - STABLE VERSION"""
    def __init__(self, n_ffts=[512, 1024, 2048], alpha=1.0, beta=1.0, gamma=1.0, use_triton=False):
        super().__init__()
        self.n_ffts = n_ffts
        self.alpha = alpha  # magnitude loss weight
        self.beta = beta    # phase loss weight  
        self.gamma = gamma  # spectral convergence weight
        # Always use PyTorch implementation
        self.use_triton = False
        
        print(f"✅ SpectralLoss: Using stable PyTorch implementation")
        
    def spectral_convergence_loss(self, pred_stft, target_stft):
        """Spectral convergence loss for better reconstruction"""
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        
        numerator = torch.norm(target_mag - pred_mag, p='fro')
        denominator = torch.norm(target_mag, p='fro')
        
        return numerator / (denominator + 1e-8)
        
    def forward(self, pred_audio, target_audio):
        total_loss = 0.0
        
        for n_fft in self.n_ffts:
            hop_length = n_fft // 4
            
            # Compute STFTs
            pred_stft = stft_transform(pred_audio, n_fft=n_fft, hop_length=hop_length)
            target_stft = stft_transform(target_audio, n_fft=n_fft, hop_length=hop_length)
            
            # Magnitude loss
            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)
            mag_loss = F.l1_loss(pred_mag, target_mag)
            
            # Spectral convergence loss
            sc_loss = self.spectral_convergence_loss(pred_stft, target_stft)
            
            # Phase loss (only where magnitude is significant)
            magnitude_mask = target_mag > 0.01 * target_mag.max()
            if magnitude_mask.any():
                pred_phase = torch.angle(pred_stft)
                target_phase = torch.angle(target_stft)
                
                # Use cosine distance for phase (better than MSE)
                phase_diff_cos = torch.cos(pred_phase - target_phase)
                phase_loss = 1.0 - phase_diff_cos[magnitude_mask].mean()
            else:
                phase_loss = torch.tensor(0.0, device=pred_audio.device)
                
            total_loss += self.alpha * mag_loss + self.beta * phase_loss + self.gamma * sc_loss
            
        return total_loss / len(self.n_ffts)

def apply_window_function(signal, window_type='hann', fade_in=True, fade_out=True):
    """Apply window function to signal for smooth boundaries"""
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

def gammatone_filterbank(spectrum, n_filters=64, use_triton=False):
    """Convenience function for gammatone filtering - always uses PyTorch"""
    filterbank = GammatoneFilterbank(n_filters=n_filters, use_triton=False)
    return filterbank(spectrum)

def get_optimal_pin_memory():
    """Get optimal pin_memory setting based on hardware"""
    return torch.cuda.is_available()

# Initialize spreading matrix cache
_psychoacoustic_masking_pytorch._spreading_cache = None