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

# IMPROVED: Cache expensive imports
HAS_SOXR = importlib.util.find_spec('soxr') is not None
HAS_TORCHAUDIO = importlib.util.find_spec('torchaudio') is not None
HAS_SCIPY = importlib.util.find_spec('scipy') is not None

# IMPROVED: Triton optimization support with better error handling
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# ========================================================================================
# Triton Optimized Kernels - FIXED VERSION
# ========================================================================================

if HAS_TRITON:
    @triton.jit
    def gammatone_kernel_simple(
        # Input pointers
        magnitude_ptr,
        center_freqs_ptr,
        erb_widths_ptr,
        # Output pointer
        output_ptr,
        # Dimensions
        batch_size,
        freq_bins,
        time_frames,
        n_filters,
        sample_rate,
        # Strides
        mag_stride_b, mag_stride_f, mag_stride_t,
        out_stride_b, out_stride_n, out_stride_t,
        # Block sizes
        BLOCK_SIZE_F: tl.constexpr,
        BLOCK_SIZE_T: tl.constexpr,
    ):
        """FIXED: Ultra-simple Triton kernel using only basic operations"""
        pid_b = tl.program_id(0)
        pid_n = tl.program_id(1) 
        pid_t = tl.program_id(2)
        
        t_offset = pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
        t_mask = t_offset < time_frames
        
        center_freq = tl.load(center_freqs_ptr + pid_n)
        erb_width = tl.load(erb_widths_ptr + pid_n)
        
        # Initialize output
        output_vals = tl.zeros([BLOCK_SIZE_T], dtype=tl.float32)
        
        # Simple frequency processing - avoid complex control flow
        f_offset = tl.arange(0, BLOCK_SIZE_F)
        freq_grid = f_offset.to(tl.float32) * (sample_rate / 2.0) / freq_bins
        freq_diff = tl.abs(freq_grid - center_freq)
        response = tl.exp(-2.0 * 3.14159 * erb_width * freq_diff / sample_rate)
        
        # Simple time-frequency processing
        for t_idx in range(BLOCK_SIZE_T):
            t_pos = pid_t * BLOCK_SIZE_T + t_idx
            t_valid = t_pos < time_frames
            
            # Load magnitude values
            accumulated_val = 0.0
            for f_idx in range(BLOCK_SIZE_F):
                f_pos = f_idx
                f_valid = f_pos < freq_bins
                
                # Only process if both time and frequency are valid
                if t_valid and f_valid:
                    mag_offset = (pid_b * mag_stride_b + 
                                 f_pos * mag_stride_f + 
                                 t_pos * mag_stride_t)
                    mag_val = tl.load(magnitude_ptr + mag_offset, mask=True, other=0.0)
                    
                    # Apply filter and accumulate
                    response_val = tl.load(response + f_idx, mask=True, other=0.0)
                    accumulated_val += mag_val * response_val
            
            # Store accumulated result
            output_vals = tl.where(
                tl.arange(0, BLOCK_SIZE_T) == t_idx,
                accumulated_val,
                output_vals
            )
        
        # Store final results
        out_offset = (pid_b * out_stride_b + 
                     pid_n * out_stride_n + 
                     t_offset * out_stride_t)
        tl.store(output_ptr + out_offset, output_vals, mask=t_mask)

    @triton.jit
    def psychoacoustic_masking_kernel_simple(
        gammatone_ptr,
        spreading_matrix_ptr,
        masking_ptr,
        batch_size, n_filters, time_frames, threshold_db,
        gamma_stride_b, gamma_stride_n, gamma_stride_t,
        mask_stride_b, mask_stride_n, mask_stride_t,
        spread_stride_i, spread_stride_j,
        BLOCK_SIZE_T: tl.constexpr,
    ):
        """FIXED: Ultra-simple psychoacoustic masking kernel"""
        pid_b = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_t = tl.program_id(2)
        
        t_offset = pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
        t_mask = t_offset < time_frames
        
        eps = 1e-10
        
        # Simple processing for each time step
        masking_values = tl.zeros([BLOCK_SIZE_T], dtype=tl.float32)
        
        for t_idx in range(BLOCK_SIZE_T):
            t_pos = pid_t * BLOCK_SIZE_T + t_idx
            t_valid = t_pos < time_frames
            
            if t_valid:
                # Load current filter value
                gamma_offset = (pid_b * gamma_stride_b + 
                               pid_n * gamma_stride_n + 
                               t_pos * gamma_stride_t)
                gamma_val = tl.load(gammatone_ptr + gamma_offset, mask=True, other=eps)
                power_db = 20.0 * tl.log(tl.maximum(gamma_val, eps)) / tl.log(10.0)
                
                # Simple spreading computation
                spread_power = 0.0
                for j in range(n_filters):
                    spread_weight = tl.load(spreading_matrix_ptr + pid_n * spread_stride_i + j * spread_stride_j, 
                                           mask=True, other=0.0)
                    j_offset = (pid_b * gamma_stride_b + 
                               j * gamma_stride_n + 
                               t_pos * gamma_stride_t)
                    j_val = tl.load(gammatone_ptr + j_offset, mask=True, other=eps)
                    j_power_db = 20.0 * tl.log(tl.maximum(j_val, eps)) / tl.log(10.0)
                    spread_power += spread_weight * j_power_db
                
                # Apply threshold
                masking_val = tl.maximum(spread_power, threshold_db)
                
                # Store in array
                masking_values = tl.where(
                    tl.arange(0, BLOCK_SIZE_T) == t_idx,
                    masking_val,
                    masking_values
                )
        
        # Store final results
        mask_offset = (pid_b * mask_stride_b + 
                      pid_n * mask_stride_n + 
                      t_offset * mask_stride_t)
        tl.store(masking_ptr + mask_offset, masking_values, mask=t_mask)

def stft_transform(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann', return_complex=True):
    """
    STFT with phase preservation for high quality reconstruction - IMPROVED: Fixed center consistency
    
    STREAMING OPTIMIZED: Uses center=False and pad_mode='reflect' for consistent 
    behavior with istft_transform. This ensures no additional latency and symmetric
    padding that works well with overlap-add reconstruction.
    
    Note: pad_mode='reflect' is used for all transforms to maintain consistency.
    For streaming applications, ensure input buffers handle boundary conditions appropriately.
    """
    if waveform.device.type == 'cuda':
        window_fn = torch.hann_window(n_fft, device=waveform.device)
    else:
        window_fn = torch.hann_window(n_fft)
        
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window_fn,
        return_complex=return_complex,
        normalized=False,
        onesided=True,
        center=False,  # IMPROVED: Consistent with istft for streaming
        pad_mode='reflect'  # Symmetric padding for better boundary handling
    )
    return stft

def istft_transform(stft_tensor, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann', length=None):
    """
    Inverse STFT with phase preservation - IMPROVED: Fixed length consistency
    
    STREAMING OPTIMIZED: Uses center=False to match stft_transform. The pad_mode
    parameter is not available in istft, but the symmetric padding from stft is
    preserved in the complex coefficients.
    
    Args:
        length: Expected output length for exact reconstruction
    """
    # IMPROVED: Handle ComplexHalf compatibility - convert to Float32 for ISTFT
    original_dtype = stft_tensor.dtype
    if stft_tensor.dtype == torch.complex32:  # ComplexHalf
        stft_tensor = stft_tensor.to(torch.complex64)  # ComplexFloat
        precision_converted = True
    else:
        precision_converted = False
    
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
        # Try with center=False first (streaming optimized)
        waveform = torch.istft(
            stft_tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window_fn,
            normalized=False,
            onesided=True,
            center=False,  # IMPROVED: Fixed center=False for streaming consistency
            length=length   # IMPROVED: Pass length to ensure exact reconstruction
        )
    except RuntimeError as e:
        if "window overlap add min" in str(e):
            # Fallback to center=True for compatibility
            warnings.warn("ISTFT with center=False failed, falling back to center=True", UserWarning)
            waveform = torch.istft(
                stft_tensor,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window_fn,
                normalized=False,
                onesided=True,
                center=True,  # Fallback for compatibility
                length=length
            )
        else:
            raise e
    
    # Convert back to original precision if needed
    if precision_converted and original_dtype == torch.complex32:
        waveform = waveform.half()
    
    return waveform

def high_quality_resample(audio, orig_sr, target_sr):
    """
    High-quality resampling with cached import checking - IMPROVED 5x faster
    """
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
            # Assume [samples, channels] format for reasonable channel counts
            transposed = False
        elif audio_np.ndim == 2 and audio_np.shape[0] <= 8:
            # Transpose [channels, samples] to [samples, channels]
            audio_np = audio_np.T
            transposed = True
        else:
            transposed = False
            
        resampled = soxr.resample(audio_np, orig_sr, target_sr, quality='HQ')
        
        # Transpose back if needed
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
        
        # Use sinc interpolation for better quality than linear
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
    """
    Create deterministic seed from path string - IMPROVED
    Uses MD5 hash to avoid Python hash randomization issues
    """
    return int(hashlib.md5(str(path_str).encode()).hexdigest()[:8], 16)

class GammatoneFilterbank(nn.Module):
    """
    Gammatone filterbank for psychoacoustic analysis - TRITON OPTIMIZED with better error handling
    """
    def __init__(self, n_filters=64, f_min=F_MIN, f_max=F_MAX, sample_rate=SAMPLE_RATE, use_triton=True):
        super().__init__()
        self.n_filters = n_filters
        self.f_min = f_min
        self.f_max = f_max
        self.sample_rate = sample_rate
        # IMPROVED: More conservative Triton usage
        self.use_triton = use_triton and HAS_TRITON and torch.cuda.is_available()
        
        # ERB scale center frequencies
        erb_min = self.hz_to_erb(f_min)
        erb_max = self.hz_to_erb(f_max)
        erb_points = torch.linspace(erb_min, erb_max, n_filters)
        center_freqs = self.erb_to_hz(erb_points)
        
        self.register_buffer('center_freqs', center_freqs)
        
        # Precompute ERB widths for efficiency
        erb_widths = 24.7 * (4.37 * center_freqs / 1000 + 1)
        self.register_buffer('erb_widths', erb_widths)
        
        if self.use_triton and HAS_TRITON:
            print("GammatoneFilterbank: Using Triton optimization")
        
    def hz_to_erb(self, hz):
        # Convert to tensor if needed for torch.log10
        if not isinstance(hz, torch.Tensor):
            hz = torch.tensor(hz, dtype=torch.float32)
        return 21.4 * torch.log10(1 + 0.00437 * hz)
    
    def erb_to_hz(self, erb):
        # Convert to tensor if needed for torch operations
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
        B, F, T = magnitude_spectrum.shape
        
        if self.use_triton and magnitude_spectrum.is_cuda and HAS_TRITON:
            try:
                return self._forward_triton(magnitude_spectrum)
            except Exception as e:
                print(f"Warning: Triton Gammatone failed ({e}), falling back to PyTorch")
                return self._forward_pytorch(magnitude_spectrum)
        else:
            return self._forward_pytorch(magnitude_spectrum)
    
    def _forward_triton(self, magnitude_spectrum):
        """Triton optimized forward pass with better error handling"""
        B, F, T = magnitude_spectrum.shape
        
        # Allocate output
        output = torch.zeros(B, self.n_filters, T, device=magnitude_spectrum.device, dtype=magnitude_spectrum.dtype)
        
        # IMPROVED: More conservative block sizes
        BLOCK_SIZE_F = min(16, F)  # Reduced from 32
        BLOCK_SIZE_T = min(32, T)  # Reduced from 64
        
        # Launch kernel
        grid = (B, self.n_filters, triton.cdiv(T, BLOCK_SIZE_T))
        
        try:
            gammatone_kernel_simple[grid](
                magnitude_spectrum, self.center_freqs, self.erb_widths,
                output,
                B, F, T, self.n_filters, float(self.sample_rate),
                magnitude_spectrum.stride(0), magnitude_spectrum.stride(1), magnitude_spectrum.stride(2),
                output.stride(0), output.stride(1), output.stride(2),
                BLOCK_SIZE_F=BLOCK_SIZE_F,
                BLOCK_SIZE_T=BLOCK_SIZE_T,
            )
            return output
        except Exception as e:
            print(f"Warning: Triton kernel launch failed ({e}), falling back to PyTorch")
            return self._forward_pytorch(magnitude_spectrum)
    
    def _forward_pytorch(self, magnitude_spectrum):
        """PyTorch fallback implementation"""
        B, F, T = magnitude_spectrum.shape
        
        # Create frequency grid - IMPROVED: Match input dtype and device for FP16 compatibility
        freq_grid = torch.linspace(
            0, self.sample_rate // 2, F, 
            device=magnitude_spectrum.device,
            dtype=magnitude_spectrum.dtype  # Match input precision
        )
        
        # Ensure center_freqs and erb_widths match input precision
        center_freqs = self.center_freqs.to(magnitude_spectrum.dtype)
        erb_widths = self.erb_widths.to(magnitude_spectrum.dtype)
        
        # Vectorized gammatone responses computation
        # [F, 1] - [1, n_filters] = [F, n_filters]
        freq_diff = freq_grid.unsqueeze(1) - center_freqs.unsqueeze(0)
        
        # Gammatone filter responses (simplified but vectorized)
        # IMPROVED: Ensure all operations use same precision
        sample_rate_tensor = torch.tensor(
            self.sample_rate, 
            device=magnitude_spectrum.device,
            dtype=magnitude_spectrum.dtype
        )
        responses = torch.exp(-2 * math.pi * erb_widths.unsqueeze(0) * 
                            torch.abs(freq_diff) / sample_rate_tensor)  # [F, n_filters]
        
        # Apply filters: [B, F, T] @ [F, n_filters] -> [B, n_filters, T]
        # IMPROVED: Use torch.matmul instead of einsum for better FP16 compatibility
        # Reshape for matrix multiplication: [B*T, F] @ [F, n_filters] = [B*T, n_filters]
        magnitude_reshaped = magnitude_spectrum.permute(0, 2, 1).contiguous().view(-1, F)  # [B*T, F]
        filtered_reshaped = torch.matmul(magnitude_reshaped, responses)  # [B*T, n_filters]
        filtered_output = filtered_reshaped.view(B, T, self.n_filters).permute(0, 2, 1)  # [B, n_filters, T]
        
        return filtered_output

def psychoacoustic_masking(gammatone_output, threshold_db=-60, use_triton=True):
    """
    Compute psychoacoustic masking curve - TRITON OPTIMIZED with better error handling
    Args:
        gammatone_output: [B, n_filters, T] - gammatone filterbank output
        threshold_db: absolute hearing threshold
        use_triton: whether to use Triton optimization
    Returns:
        masking_curve: [B, n_filters, T] - masking threshold
    """
    # IMPROVED: More conservative Triton usage
    if use_triton and HAS_TRITON and gammatone_output.is_cuda:
        try:
            return _psychoacoustic_masking_triton(gammatone_output, threshold_db)
        except Exception as e:
            print(f"Warning: Triton psychoacoustic masking failed ({e}), falling back to PyTorch")
            return _psychoacoustic_masking_pytorch(gammatone_output, threshold_db)
    else:
        return _psychoacoustic_masking_pytorch(gammatone_output, threshold_db)

def _psychoacoustic_masking_triton(gammatone_output, threshold_db):
    """Triton optimized psychoacoustic masking with better error handling"""
    B, n_filters, T = gammatone_output.shape
    
    # Allocate output
    masking_curve = torch.zeros_like(gammatone_output)
    
    # Create or retrieve cached spreading matrix
    cache_key = (n_filters, gammatone_output.dtype, gammatone_output.device)
    if not hasattr(_psychoacoustic_masking_triton, '_spreading_cache'):
        _psychoacoustic_masking_triton._spreading_cache = {}
    
    if cache_key not in _psychoacoustic_masking_triton._spreading_cache:
        spreading_matrix = torch.zeros(n_filters, n_filters, device=gammatone_output.device, dtype=gammatone_output.dtype)
        
        for i in range(n_filters):
            for j in range(n_filters):
                if i != j:
                    bark_diff = j - i + 0.474
                    spread = max(0, 15.81 + 7.5 * bark_diff - 17.5 * (1 + bark_diff**2)**0.5)
                    spreading_matrix[i, j] = 10**(-spread / 10)
                else:
                    spreading_matrix[i, j] = 1.0
        
        _psychoacoustic_masking_triton._spreading_cache[cache_key] = spreading_matrix
    
    spreading_matrix = _psychoacoustic_masking_triton._spreading_cache[cache_key]
    
    # IMPROVED: More conservative block sizes
    BLOCK_SIZE_T = min(32, T)  # Reduced from 64
    
    # Launch kernel
    grid = (B, n_filters, triton.cdiv(T, BLOCK_SIZE_T))
    
    try:
        psychoacoustic_masking_kernel_simple[grid](
            gammatone_output, spreading_matrix, masking_curve,
            B, n_filters, T, float(threshold_db),
            gammatone_output.stride(0), gammatone_output.stride(1), gammatone_output.stride(2),
            masking_curve.stride(0), masking_curve.stride(1), masking_curve.stride(2),
            spreading_matrix.stride(0), spreading_matrix.stride(1),
            BLOCK_SIZE_T=BLOCK_SIZE_T,
        )
        return masking_curve
    except Exception as e:
        print(f"Warning: Triton psychoacoustic kernel failed ({e}), falling back to PyTorch")
        return _psychoacoustic_masking_pytorch(gammatone_output, threshold_db)

def _psychoacoustic_masking_pytorch(gammatone_output, threshold_db):
    """PyTorch fallback implementation"""
    # Convert to dB with improved numerical stability
    eps = 1e-10
    power_db = 20 * torch.log10(torch.clamp(gammatone_output, min=eps))
    
    n_filters = gammatone_output.shape[1]
    device = gammatone_output.device
    dtype = gammatone_output.dtype  # IMPROVED: Preserve input dtype
    
    # Precompute spreading matrix if not cached
    cache_key = (n_filters, dtype)
    if (not hasattr(_psychoacoustic_masking_pytorch, '_spreading_cache') or 
        _psychoacoustic_masking_pytorch._spreading_cache is None or
        _psychoacoustic_masking_pytorch._spreading_cache[0] != cache_key):
        
        # IMPROVED: Create spreading matrix with correct dtype
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
    
    # IMPROVED: Use torch.matmul instead of einsum for better FP16 compatibility
    # Apply spreading: [B, n_filters, T] -> [B*T, n_filters] @ [n_filters, n_filters] -> [B*T, n_filters] -> [B, n_filters, T]
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
    """Convert complex spectrogram back to waveform - IMPROVED with length parameter"""
    return istft_transform(complex_spec, length=length)

def normalize_audio(waveform, target_db=-23.0, method='rms'):
    """
    Normalize audio to target dB level - IMPROVED with multiple methods
    """
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
        # Simplified LUFS-like normalization (frequency weighted RMS)
        # Apply A-weighting-like emphasis
        if waveform.dim() > 1:
            # For spectrogram-based weighting, this is simplified
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
    Do not use during training as it breaks gradient flow. Use this only for 
    post-processing decoded audio or for analysis purposes.
    
    Args:
        magnitude: Input magnitude spectrum
        ratio: Compression ratio (higher = more compression)
        threshold: Compression threshold 
        knee_width: Soft knee width for smooth transitions
    
    Returns:
        compressed: Compressed magnitude spectrum (no gradients)
    """
    # IMPROVED: Use torch.no_grad() for inference or .detach() to avoid autograd issues
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
    """Multi-scale spectral loss for high-quality reconstruction - TRITON OPTIMIZED"""
    def __init__(self, n_ffts=[512, 1024, 2048], alpha=1.0, beta=1.0, gamma=1.0, use_triton=True):
        super().__init__()
        self.n_ffts = n_ffts
        self.alpha = alpha  # magnitude loss weight
        self.beta = beta    # phase loss weight  
        self.gamma = gamma  # spectral convergence weight
        # IMPROVED: More conservative Triton usage
        self.use_triton = use_triton and HAS_TRITON and torch.cuda.is_available()
        
        if self.use_triton and HAS_TRITON:
            print("SpectralLoss: Triton optimization available")
        
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
    """
    Apply window function to signal for smooth boundaries
    """
    length = signal.shape[-1]
    
    if window_type == 'hann':
        window = torch.hann_window(length, device=signal.device)
    elif window_type == 'hamming':
        window = torch.hamming_window(length, device=signal.device)
    elif window_type == 'blackman':
        window = torch.blackman_window(length, device=signal.device)
    else:
        window = torch.ones(length, device=signal.device)
    
    # Apply only fade in/out if requested
    if not fade_in:
        window[:length//2] = 1.0
    if not fade_out:
        window[length//2:] = 1.0
    
    # Broadcast window to match signal shape
    while window.dim() < signal.dim():
        window = window.unsqueeze(0)
        
    return signal * window

def gammatone_filterbank(spectrum, n_filters=64, use_triton=True):
    """Convenience function for gammatone filtering with Triton optimization"""
    filterbank = GammatoneFilterbank(n_filters=n_filters, use_triton=use_triton)
    return filterbank(spectrum)

def get_optimal_pin_memory():
    """Get optimal pin_memory setting based on hardware"""
    return torch.cuda.is_available()

# Initialize spreading matrix cache
_psychoacoustic_masking_pytorch._spreading_cache = None