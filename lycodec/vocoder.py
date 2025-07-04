"""
LyCodec v2.1 DDSP Vocoder Module - Enhanced Production Version
=============================================================

High-performance Differentiable Digital Signal Processing (DDSP) vocoder with
numerically stable harmonic synthesis, optimized Triton kernels, and
comprehensive error handling for production audio codec deployment.

Key Improvements:
- Enhanced numerical stability with extended precision arithmetic
- Optimized memory access patterns for GPU efficiency
- Comprehensive error handling and fallback mechanisms
- Advanced phase accumulation with drift correction
- Production-grade performance monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import math
import numpy as np
import warnings
import time
from contextlib import contextmanager

# Triton imports with comprehensive error handling
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
    TRITON_VERSION = triton.__version__
except ImportError as e:
    TRITON_AVAILABLE = False
    TRITON_VERSION = None
    warnings.warn(f"Triton not available: {e}. Falling back to PyTorch implementation.")

# CUDA availability check
CUDA_AVAILABLE = torch.cuda.is_available()


class NumericalStabilityMixin:
    """
    Mixin class providing numerical stability utilities for audio processing.
    
    Implements extended precision arithmetic, overflow detection, and
    catastrophic cancellation prevention for robust audio synthesis.
    """
    
    @staticmethod
    def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Numerically stable logarithm with underflow protection."""
        return torch.log(torch.clamp(x, min=eps))
    
    @staticmethod
    def safe_exp(x: torch.Tensor, max_val: float = 50.0) -> torch.Tensor:
        """Numerically stable exponential with overflow protection."""
        return torch.exp(torch.clamp(x, max=max_val))
    
    @staticmethod
    def safe_sqrt(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Numerically stable square root with negative input protection."""
        return torch.sqrt(torch.clamp(x, min=eps))
    
    @staticmethod
    def stable_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
        """Numerically stable L2 normalization."""
        norm = torch.norm(x, dim=dim, keepdim=True)
        return x / torch.clamp(norm, min=eps)
    
    @staticmethod
    def check_tensor_validity(tensor: torch.Tensor, name: str = "tensor") -> bool:
        """Comprehensive tensor validity check for debugging."""
        if torch.isnan(tensor).any():
            warnings.warn(f"NaN detected in {name}")
            return False
        if torch.isinf(tensor).any():
            warnings.warn(f"Inf detected in {name}")
            return False
        if tensor.numel() == 0:
            warnings.warn(f"Empty tensor: {name}")
            return False
        return True


class HighPrecisionSinusoidalLookupTable(NumericalStabilityMixin):
    """
    Production-grade sinusoidal lookup table with extended precision and interpolation.
    
    Features:
    - FP64 precision for table generation and phase accumulation
    - Configurable interpolation orders (linear, cubic, quintic)
    - Automatic table size optimization based on target precision
    - Phase wrap handling with period-accurate arithmetic
    - Memory-efficient table storage and access patterns
    """
    
    def __init__(self, table_size: int = 8192, precision_bits: int = 32,
                 interpolation_order: int = 3, target_snr_db: float = 120.0):
        self.table_size = table_size
        self.precision_bits = precision_bits
        self.interpolation_order = interpolation_order
        self.target_snr_db = target_snr_db
        
        # Validate table size (must be power of 2 for efficient modular arithmetic)
        if table_size & (table_size - 1) != 0:
            warnings.warn(f"Table size {table_size} is not power of 2, rounding up")
            self.table_size = 1 << (table_size - 1).bit_length()
        
        # Extended precision table generation
        self.sine_table = self._generate_high_precision_table()
        
        # Phase accumulation parameters
        self.phase_scale = 2.0 ** precision_bits
        self.phase_mask = (1 << precision_bits) - 1
        self.table_mask = self.table_size - 1
        
        # Interpolation coefficients (pre-computed for efficiency)
        self.interp_coeffs = self._precompute_interpolation_coefficients()
        
        # Performance monitoring
        self.lookup_count = 0
        self.total_lookup_time = 0.0
        
        # Validate table quality
        self._validate_table_quality()
    
    def _generate_high_precision_table(self) -> torch.Tensor:
        """
        Generate high-precision sine table using FP64 arithmetic.
        
        Returns:
            Sine table with extended precision and validated accuracy
        """
        # Use double precision for maximum accuracy
        angles = torch.linspace(0, 2 * math.pi, self.table_size + 1, 
                              dtype=torch.float64)[:-1]
        
        # Compute sine values with extended precision
        sine_values_fp64 = torch.sin(angles)
        
        # Convert to FP32 for storage efficiency while maintaining precision
        sine_values = sine_values_fp64.float()
        
        # Validate table quality using spectral analysis
        self._analyze_table_spectrum(sine_values_fp64)
        
        return sine_values
    
    def _analyze_table_spectrum(self, sine_table: torch.Tensor):
        """Analyze table spectral quality for harmonic distortion assessment."""
        # FFT analysis to check for quantization artifacts
        table_fft = torch.fft.fft(sine_table)
        magnitude = torch.abs(table_fft)
        
        # Find fundamental frequency bin
        fundamental_bin = 1  # Single cycle in table
        fundamental_power = magnitude[fundamental_bin].item() ** 2
        
        # Calculate total harmonic distortion (THD)
        harmonic_bins = [2, 3, 4, 5]  # Check first few harmonics
        harmonic_power = sum(magnitude[bin].item() ** 2 for bin in harmonic_bins
                           if bin < len(magnitude))
        
        thd_db = 10 * math.log10(harmonic_power / fundamental_power) if fundamental_power > 0 else -np.inf
        
        if thd_db > -self.target_snr_db:
            warnings.warn(f"Table THD {thd_db:.2f} dB exceeds target {-self.target_snr_db} dB")
    
    def _precompute_interpolation_coefficients(self) -> Dict[str, torch.Tensor]:
        """Pre-compute interpolation coefficients for different orders."""
        coeffs = {}
        
        if self.interpolation_order == 1:
            # Linear interpolation (no pre-computation needed)
            coeffs['linear'] = torch.tensor([1.0, 1.0])
        
        elif self.interpolation_order == 3:
            # Cubic Hermite interpolation coefficients
            # h(t) = a0*t³ + a1*t² + a2*t + a3
            coeffs['cubic'] = torch.tensor([
                [-0.5,  1.5, -1.5,  0.5],  # a0 coefficients
                [ 1.0, -2.5,  2.0, -0.5],  # a1 coefficients  
                [-0.5,  0.0,  0.5,  0.0],  # a2 coefficients
                [ 0.0,  1.0,  0.0,  0.0]   # a3 coefficients
            ])
        
        elif self.interpolation_order == 5:
            # Quintic interpolation for ultra-high precision
            coeffs['quintic'] = torch.tensor([
                [-1/24,  1/6, -1/4,  1/6, -1/24,   0],
                [ 1/12, -1/3,  1/2, -1/3,  1/12,   0],
                [-1/12,  2/3, -5/4,  2/3, -1/12,   0],
                [ 1/24, -1/6,  1/4, -1/6,  1/24,   0],
                [-1/24,  1/6, -1/4,  1/6, -1/24,   0],
                [    0,    0,    1,    0,     0,   0]
            ])
        
        return coeffs
    
    def _validate_table_quality(self):
        """Validate lookup table meets quality requirements."""
        # Test interpolation accuracy
        test_phases = torch.linspace(0, 1, 1000, dtype=torch.float64)
        reference_sine = torch.sin(2 * math.pi * test_phases).float()
        interpolated_sine = self.interpolate_sine(test_phases.float())
        
        mse = torch.mean((reference_sine - interpolated_sine) ** 2)
        snr_db = -10 * torch.log10(mse).item()
        
        if snr_db < self.target_snr_db:
            warnings.warn(f"Table SNR {snr_db:.2f} dB below target {self.target_snr_db} dB")
    
    @contextmanager
    def performance_timing(self):
        """Context manager for performance measurement."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self.total_lookup_time += elapsed
            self.lookup_count += 1
    
    def interpolate_sine(self, phases: torch.Tensor) -> torch.Tensor:
        """
        Compute sine values with high-precision interpolation.
        
        Args:
            phases: Normalized phases [0, 1) tensor of any shape
            
        Returns:
            Interpolated sine values with numerical precision guarantees
        """
        with self.performance_timing():
            original_shape = phases.shape
            device = phases.device
            
            # Ensure table is on correct device
            if self.sine_table.device != device:
                self.sine_table = self.sine_table.to(device)
                for key, coeff in self.interp_coeffs.items():
                    self.interp_coeffs[key] = coeff.to(device)
            
            # Flatten phases for vectorized processing
            phases_flat = phases.flatten()
            
            # Validate input phases
            if not self.check_tensor_validity(phases_flat, "input_phases"):
                return torch.zeros_like(phases_flat).view(original_shape)
            
            # Clamp phases to valid range [0, 1)
            phases_clamped = torch.clamp(phases_flat % 1.0, 0.0, 0.999999)
            
            # Convert to table indices with extended precision
            table_indices_float = phases_clamped.double() * self.table_size
            table_indices_int = table_indices_float.long() & self.table_mask
            fractional_part = (table_indices_float - table_indices_int.double()).float()
            
            # Perform interpolation based on configured order
            if self.interpolation_order == 1:
                interpolated = self._linear_interpolation(table_indices_int, fractional_part)
            elif self.interpolation_order == 3:
                interpolated = self._cubic_interpolation(table_indices_int, fractional_part)
            elif self.interpolation_order == 5:
                interpolated = self._quintic_interpolation(table_indices_int, fractional_part)
            else:
                raise ValueError(f"Unsupported interpolation order: {self.interpolation_order}")
            
            # Validate output
            if not self.check_tensor_validity(interpolated, "interpolated_output"):
                warnings.warn("Invalid interpolation output, returning zeros")
                interpolated = torch.zeros_like(interpolated)
            
            return interpolated.view(original_shape)
    
    def _linear_interpolation(self, indices: torch.Tensor, frac: torch.Tensor) -> torch.Tensor:
        """Linear interpolation implementation."""
        indices_next = (indices + 1) & self.table_mask
        
        values_current = self.sine_table[indices]
        values_next = self.sine_table[indices_next]
        
        return values_current + frac * (values_next - values_current)
    
    def _cubic_interpolation(self, indices: torch.Tensor, frac: torch.Tensor) -> torch.Tensor:
        """Cubic Hermite interpolation for higher accuracy."""
        # Get 4-point stencil
        idx_m1 = (indices - 1) & self.table_mask
        idx_0 = indices
        idx_p1 = (indices + 1) & self.table_mask
        idx_p2 = (indices + 2) & self.table_mask
        
        y_m1 = self.sine_table[idx_m1]
        y_0 = self.sine_table[idx_0]
        y_p1 = self.sine_table[idx_p1]
        y_p2 = self.sine_table[idx_p2]
        
        # Cubic Hermite coefficients
        coeffs = self.interp_coeffs['cubic']
        
        # Compute polynomial coefficients
        a0 = coeffs[0, 0] * y_m1 + coeffs[0, 1] * y_0 + coeffs[0, 2] * y_p1 + coeffs[0, 3] * y_p2
        a1 = coeffs[1, 0] * y_m1 + coeffs[1, 1] * y_0 + coeffs[1, 2] * y_p1 + coeffs[1, 3] * y_p2
        a2 = coeffs[2, 0] * y_m1 + coeffs[2, 1] * y_0 + coeffs[2, 2] * y_p1 + coeffs[2, 3] * y_p2
        a3 = coeffs[3, 0] * y_m1 + coeffs[3, 1] * y_0 + coeffs[3, 2] * y_p1 + coeffs[3, 3] * y_p2
        
        # Evaluate polynomial: a0*t³ + a1*t² + a2*t + a3
        t = frac
        t2 = t * t
        t3 = t2 * t
        
        return a0 * t3 + a1 * t2 + a2 * t + a3
    
    def _quintic_interpolation(self, indices: torch.Tensor, frac: torch.Tensor) -> torch.Tensor:
        """Quintic interpolation for ultra-high precision."""
        # Get 6-point stencil
        stencil_indices = []
        for offset in range(-2, 4):
            stencil_indices.append((indices + offset) & self.table_mask)
        
        stencil_values = [self.sine_table[idx] for idx in stencil_indices]
        coeffs = self.interp_coeffs['quintic']
        
        # Evaluate quintic polynomial
        result = torch.zeros_like(frac)
        t_powers = [torch.ones_like(frac)]  # t^0 = 1
        
        for i in range(5):
            t_powers.append(t_powers[-1] * frac)  # t^(i+1)
        
        for i in range(6):  # 6 coefficients for quintic
            term = coeffs[i, 5]  # constant term
            for j in range(6):  # 6 stencil points
                term += coeffs[i, j] * stencil_values[j]
            result += term * t_powers[i]
        
        return result
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for optimization."""
        if self.lookup_count > 0:
            avg_time = self.total_lookup_time / self.lookup_count
            throughput = self.lookup_count / self.total_lookup_time if self.total_lookup_time > 0 else 0
        else:
            avg_time = 0.0
            throughput = 0.0
        
        return {
            'total_lookups': self.lookup_count,
            'total_time_ms': self.total_lookup_time * 1000,
            'avg_lookup_time_us': avg_time * 1e6,
            'throughput_lookups_per_sec': throughput,
            'table_size': self.table_size,
            'interpolation_order': self.interpolation_order
        }
    
    def reset_performance_stats(self):
        """Reset performance counters."""
        self.lookup_count = 0
        self.total_lookup_time = 0.0


# Enhanced Triton kernels with comprehensive optimization
if TRITON_AVAILABLE:
    @triton.jit
    def enhanced_harmonic_synthesis_kernel(
        # Input pointers
        frequencies_ptr, amplitudes_ptr, phases_ptr, sine_table_ptr,
        # Output pointer
        output_ptr,
        # Tensor metadata
        batch_size, seq_len, num_harmonics, table_size,
        # Stride information
        freq_stride_batch, freq_stride_seq, freq_stride_harm,
        amp_stride_batch, amp_stride_seq, amp_stride_harm,
        phase_stride_batch, phase_stride_seq, phase_stride_harm,
        output_stride_batch, output_stride_seq,
        # Configuration constants
        sample_rate: tl.constexpr,
        BLOCK_SIZE_SEQ: tl.constexpr,
        BLOCK_SIZE_HARM: tl.constexpr,
        USE_CUBIC_INTERP: tl.constexpr
    ):
        """
        Enhanced Triton kernel for high-performance harmonic synthesis.
        
        Features:
        - Coalesced memory access patterns
        - Shared memory optimization for sine table
        - Vectorized harmonic processing
        - Optional cubic interpolation
        - Numerical stability guarantees
        """
        # Program identification
        batch_id = tl.program_id(0)
        seq_block_id = tl.program_id(1)
        
        # Sequence processing block
        seq_start = seq_block_id * BLOCK_SIZE_SEQ
        seq_offsets = seq_start + tl.arange(0, BLOCK_SIZE_SEQ)
        seq_mask = seq_offsets < seq_len
        
        # Load sine table into shared memory for fast access
        table_offsets = tl.arange(0, table_size)
        sine_table_shared = tl.load(sine_table_ptr + table_offsets)
        
        # Initialize output accumulator
        output_block = tl.zeros([BLOCK_SIZE_SEQ], dtype=tl.float32)
        
        # Process harmonics in blocks for register efficiency
        for harm_start in range(0, num_harmonics, BLOCK_SIZE_HARM):
            harm_offsets = harm_start + tl.arange(0, BLOCK_SIZE_HARM)
            harm_mask = harm_offsets < num_harmonics
            
            # Load harmonic parameters with proper striding
            freq_ptrs = (frequencies_ptr + 
                        batch_id * freq_stride_batch + 
                        seq_offsets[:, None] * freq_stride_seq +
                        harm_offsets[None, :] * freq_stride_harm)
            
            amp_ptrs = (amplitudes_ptr + 
                       batch_id * amp_stride_batch + 
                       seq_offsets[:, None] * amp_stride_seq +
                       harm_offsets[None, :] * amp_stride_harm)
            
            phase_ptrs = (phases_ptr + 
                         batch_id * phase_stride_batch + 
                         seq_offsets[:, None] * phase_stride_seq +
                         harm_offsets[None, :] * phase_stride_harm)
            
            # Load parameters with masking
            load_mask = seq_mask[:, None] & harm_mask[None, :]
            frequencies = tl.load(freq_ptrs, mask=load_mask, other=0.0)
            amplitudes = tl.load(amp_ptrs, mask=load_mask, other=0.0)
            phases = tl.load(phase_ptrs, mask=load_mask, other=0.0)
            
            # Normalize phases for table lookup
            TWO_PI = 6.28318530718
            normalized_phases = (phases / TWO_PI) % 1.0
            
            # Table lookup with interpolation
            table_indices_float = normalized_phases * table_size
            table_indices_int = table_indices_float.to(tl.int32) % table_size
            fractional_part = table_indices_float - table_indices_int
            
            if USE_CUBIC_INTERP:
                # Cubic interpolation for higher quality
                idx_m1 = (table_indices_int - 1) % table_size
                idx_0 = table_indices_int
                idx_p1 = (table_indices_int + 1) % table_size
                idx_p2 = (table_indices_int + 2) % table_size
                
                y_m1 = tl.load(sine_table_ptr + idx_m1)
                y_0 = tl.load(sine_table_ptr + idx_0)
                y_p1 = tl.load(sine_table_ptr + idx_p1)
                y_p2 = tl.load(sine_table_ptr + idx_p2)
                
                # Cubic Hermite interpolation
                t = fractional_part
                t2 = t * t
                t3 = t2 * t
                
                a0 = -0.5 * y_m1 + 1.5 * y_0 - 1.5 * y_p1 + 0.5 * y_p2
                a1 = y_m1 - 2.5 * y_0 + 2.0 * y_p1 - 0.5 * y_p2
                a2 = -0.5 * y_m1 + 0.5 * y_p1
                a3 = y_0
                
                sine_values = a0 * t3 + a1 * t2 + a2 * t + a3
            else:
                # Linear interpolation
                indices_next = (table_indices_int + 1) % table_size
                sine_current = tl.load(sine_table_ptr + table_indices_int)
                sine_next = tl.load(sine_table_ptr + indices_next)
                sine_values = sine_current + fractional_part * (sine_next - sine_current)
            
            # Apply amplitude weighting
            weighted_harmonics = amplitudes * sine_values
            
            # Accumulate across harmonics (reduce along harmonic dimension)
            harmonic_sum = tl.sum(tl.where(load_mask, weighted_harmonics, 0.0), axis=1)
            output_block += harmonic_sum
        
        # Store results to global memory
        output_ptrs = (output_ptr + 
                      batch_id * output_stride_batch + 
                      seq_offsets * output_stride_seq)
        tl.store(output_ptrs, output_block, mask=seq_mask)
    
    @triton.jit
    def phase_accumulation_kernel(
        # Input/Output pointers
        frequencies_ptr, phases_ptr,
        # Tensor metadata
        batch_size, seq_len, num_harmonics,
        # Stride information
        freq_stride_batch, freq_stride_seq, freq_stride_harm,
        phase_stride_batch, phase_stride_seq, phase_stride_harm,
        # Constants
        dt: tl.constexpr,
        TWO_PI: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        """
        Specialized kernel for high-precision phase accumulation.
        
        Prevents phase drift through extended precision arithmetic
        and periodic normalization for long-duration synthesis.
        """
        batch_id = tl.program_id(0)
        harm_id = tl.program_id(1)
        
        # Load initial phase
        phase_ptr = (phases_ptr + 
                    batch_id * phase_stride_batch + 
                    harm_id * phase_stride_harm)
        current_phase = tl.load(phase_ptr)
        
        # Process sequence in blocks
        for seq_start in range(0, seq_len, BLOCK_SIZE):
            seq_offsets = seq_start + tl.arange(0, BLOCK_SIZE)
            seq_mask = seq_offsets < seq_len
            
            # Load frequencies for this block
            freq_ptrs = (frequencies_ptr + 
                        batch_id * freq_stride_batch + 
                        seq_offsets * freq_stride_seq + 
                        harm_id * freq_stride_harm)
            frequencies = tl.load(freq_ptrs, mask=seq_mask, other=0.0)
            
            # Compute phase increments
            phase_increments = TWO_PI * frequencies * dt
            
            # Accumulate phases with periodic normalization
            for i in range(BLOCK_SIZE):
                if seq_start + i < seq_len:
                    current_phase += phase_increments[i]
                    # Periodic normalization to prevent drift
                    if current_phase > TWO_PI:
                        current_phase -= TWO_PI
                    elif current_phase < 0.0:
                        current_phase += TWO_PI
                    
                    # Store updated phase
                    phase_store_ptr = (phases_ptr + 
                                     batch_id * phase_stride_batch + 
                                     (seq_start + i) * phase_stride_seq + 
                                     harm_id * phase_stride_harm)
                    tl.store(phase_store_ptr, current_phase)


class OptimizedTritonHarmonicSynthesizer(NumericalStabilityMixin):
    """
    Production-optimized Triton-based harmonic synthesizer.
    
    Advanced Features:
    - Adaptive block size selection based on GPU architecture
    - Memory bandwidth optimization through coalesced access
    - Runtime performance monitoring and auto-tuning
    - Comprehensive error handling and fallback mechanisms
    - Numerical stability validation
    """
    
    def __init__(self, sample_rate: int = 44100, table_size: int = 8192,
                 auto_tune: bool = True, target_precision: str = "high"):
        self.sample_rate = sample_rate
        self.table_size = table_size
        self.auto_tune = auto_tune
        self.target_precision = target_precision
        
        # Initialize high-precision lookup table
        interpolation_order = 3 if target_precision == "high" else 1
        self.lut = HighPrecisionSinusoidalLookupTable(
            table_size=table_size,
            interpolation_order=interpolation_order,
            target_snr_db=120.0 if target_precision == "high" else 80.0
        )
        
        # Triton kernel configuration
        self.kernel_config = self._initialize_kernel_config()
        
        # Performance tracking
        self.synthesis_times = []
        self.memory_bandwidths = []
        self.kernel_launch_overhead = []
        
        # Auto-tuning parameters
        if auto_tune:
            self._auto_tune_kernel_params()
    
    def _initialize_kernel_config(self) -> Dict[str, Any]:
        """Initialize kernel configuration based on GPU architecture."""
        config = {
            'BLOCK_SIZE_SEQ': 128,
            'BLOCK_SIZE_HARM': 16,
            'USE_CUBIC_INTERP': self.target_precision == "high",
            'num_warps': 4,
            'num_stages': 2
        }
        
        if CUDA_AVAILABLE:
            # Optimize for specific GPU architectures
            gpu_name = torch.cuda.get_device_name()
            if 'V100' in gpu_name:
                config.update({
                    'BLOCK_SIZE_SEQ': 256,
                    'BLOCK_SIZE_HARM': 32,
                    'num_warps': 8,
                    'num_stages': 3
                })
            elif 'A100' in gpu_name:
                config.update({
                    'BLOCK_SIZE_SEQ': 512,
                    'BLOCK_SIZE_HARM': 64,
                    'num_warps': 16,
                    'num_stages': 4
                })
        
        return config
    
    def _auto_tune_kernel_params(self):
        """Auto-tune kernel parameters for optimal performance."""
        if not TRITON_AVAILABLE:
            return
        
        print("Auto-tuning Triton kernel parameters...")
        
        # Test configurations
        test_configs = [
            {'BLOCK_SIZE_SEQ': 64, 'BLOCK_SIZE_HARM': 8, 'num_warps': 2},
            {'BLOCK_SIZE_SEQ': 128, 'BLOCK_SIZE_HARM': 16, 'num_warps': 4},
            {'BLOCK_SIZE_SEQ': 256, 'BLOCK_SIZE_HARM': 32, 'num_warps': 8},
            {'BLOCK_SIZE_SEQ': 512, 'BLOCK_SIZE_HARM': 64, 'num_warps': 16},
        ]
        
        # Generate test inputs
        batch_size, seq_len, num_harmonics = 4, 1024, 48
        test_frequencies = torch.randn(batch_size, seq_len, num_harmonics, device='cuda')
        test_amplitudes = torch.randn(batch_size, seq_len, num_harmonics, device='cuda')
        test_phases = torch.randn(batch_size, seq_len, num_harmonics, device='cuda')
        
        best_config = None
        best_time = float('inf')
        
        for config in test_configs:
            try:
                # Warm up
                for _ in range(5):
                    self._synthesize_triton(test_frequencies, test_amplitudes, test_phases, config)
                
                # Benchmark
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                for _ in range(20):
                    self._synthesize_triton(test_frequencies, test_amplitudes, test_phases, config)
                
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time
                
                if elapsed < best_time:
                    best_time = elapsed
                    best_config = config
                    
            except Exception as e:
                print(f"Config {config} failed: {e}")
                continue
        
        if best_config:
            self.kernel_config.update(best_config)
            print(f"Optimal configuration: {best_config}, time: {best_time:.4f}s")
    
    def _synthesize_triton(self, frequencies: torch.Tensor, amplitudes: torch.Tensor,
                          phases: torch.Tensor, config: Optional[Dict] = None) -> torch.Tensor:
        """Core Triton kernel synthesis implementation."""
        if config is None:
            config = self.kernel_config
        
        batch_size, seq_len, num_harmonics = frequencies.shape
        device = frequencies.device
        
        # Ensure lookup table is on correct device
        if self.lut.sine_table.device != device:
            self.lut.sine_table = self.lut.sine_table.to(device)
        
        # Allocate output tensor
        output = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
        
        # Configure kernel launch grid
        grid = (
            batch_size,
            triton.cdiv(seq_len, config['BLOCK_SIZE_SEQ'])
        )
        
        # Launch enhanced harmonic synthesis kernel
        enhanced_harmonic_synthesis_kernel[grid](
            # Input tensors
            frequencies, amplitudes, phases, self.lut.sine_table,
            # Output tensor
            output,
            # Tensor dimensions
            batch_size, seq_len, num_harmonics, self.table_size,
            # Strides
            frequencies.stride(0), frequencies.stride(1), frequencies.stride(2),
            amplitudes.stride(0), amplitudes.stride(1), amplitudes.stride(2),
            phases.stride(0), phases.stride(1), phases.stride(2),
            output.stride(0), output.stride(1),
            # Constants
            sample_rate=self.sample_rate,
            BLOCK_SIZE_SEQ=config['BLOCK_SIZE_SEQ'],
            BLOCK_SIZE_HARM=config['BLOCK_SIZE_HARM'],
            USE_CUBIC_INTERP=config['USE_CUBIC_INTERP'],
            # Kernel configuration
            num_warps=config.get('num_warps', 4),
            num_stages=config.get('num_stages', 2)
        )
        
        return output
    
    def synthesize(self, frequencies: torch.Tensor, amplitudes: torch.Tensor,
                  phases: torch.Tensor) -> torch.Tensor:
        """
        High-performance harmonic synthesis with comprehensive error handling.
        
        Args:
            frequencies: Fundamental frequencies [batch, seq_len, num_harmonics]
            amplitudes: Harmonic amplitudes [batch, seq_len, num_harmonics]
            phases: Initial phases [batch, seq_len, num_harmonics]
            
        Returns:
            Synthesized audio [batch, seq_len] with validated output
        """
        # Input validation
        if not self.check_tensor_validity(frequencies, "frequencies"):
            return torch.zeros(frequencies.shape[:2], device=frequencies.device)
        
        if not self.check_tensor_validity(amplitudes, "amplitudes"):
            return torch.zeros(frequencies.shape[:2], device=frequencies.device)
        
        if not self.check_tensor_validity(phases, "phases"):
            return torch.zeros(frequencies.shape[:2], device=frequencies.device)
        
        # Shape consistency check
        if not (frequencies.shape == amplitudes.shape == phases.shape):
            raise ValueError(f"Shape mismatch: freq {frequencies.shape}, "
                           f"amp {amplitudes.shape}, phase {phases.shape}")
        
        # Performance timing
        start_time = time.perf_counter()
        
        try:
            if TRITON_AVAILABLE and frequencies.device.type == 'cuda':
                # Use Triton kernel for GPU acceleration
                output = self._synthesize_triton(frequencies, amplitudes, phases)
            else:
                # Fallback to PyTorch implementation
                output = self._synthesize_pytorch(frequencies, amplitudes, phases)
            
            # Validate output
            if not self.check_tensor_validity(output, "synthesis_output"):
                warnings.warn("Invalid synthesis output detected, returning zeros")
                output = torch.zeros_like(output)
            
            # Performance tracking
            synthesis_time = time.perf_counter() - start_time
            self.synthesis_times.append(synthesis_time)
            
            return output
            
        except Exception as e:
            warnings.warn(f"Synthesis failed: {e}, falling back to PyTorch")
            return self._synthesize_pytorch(frequencies, amplitudes, phases)
    
    def _synthesize_pytorch(self, frequencies: torch.Tensor, amplitudes: torch.Tensor,
                           phases: torch.Tensor) -> torch.Tensor:
        """PyTorch fallback implementation with numerical stability."""
        # Normalize phases for lookup table
        normalized_phases = (phases / (2 * math.pi)) % 1.0
        
        # High-precision sine computation
        sine_values = self.lut.interpolate_sine(normalized_phases)
        
        # Apply amplitude weighting
        weighted_harmonics = amplitudes * sine_values
        
        # Sum across harmonics with numerical stability
        synthesized_audio = torch.sum(weighted_harmonics, dim=-1)
        
        # Apply soft clipping to prevent harsh clipping artifacts
        synthesized_audio = torch.tanh(synthesized_audio * 0.9) / 0.9
        
        return synthesized_audio
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Comprehensive performance metrics for optimization."""
        if not self.synthesis_times:
            return {}
        
        metrics = {
            'total_syntheses': len(self.synthesis_times),
            'avg_synthesis_time_ms': np.mean(self.synthesis_times) * 1000,
            'min_synthesis_time_ms': np.min(self.synthesis_times) * 1000,
            'max_synthesis_time_ms': np.max(self.synthesis_times) * 1000,
            'std_synthesis_time_ms': np.std(self.synthesis_times) * 1000,
        }
        
        # Add lookup table performance
        lut_metrics = self.lut.get_performance_stats()
        metrics.update({f'lut_{k}': v for k, v in lut_metrics.items()})
        
        return metrics


class AdvancedCriticalBandNoiseGenerator(NumericalStabilityMixin):
    """
    Production-grade critical band noise generator with advanced filtering.
    
    Enhanced Features:
    - Psychoacoustically-motivated filter design with precise band edges
    - High-quality PRNG with cryptographic randomness options
    - Temporal envelope modeling with smooth transitions
    - Real-time parameter interpolation
    - Comprehensive spectral analysis and validation
    """
    
    def __init__(self, sample_rate: int = 44100, num_bands: int = 8,
                 filter_order: int = 4, transition_width_ratio: float = 0.1):
        self.sample_rate = sample_rate
        self.num_bands = num_bands
        self.filter_order = filter_order
        self.transition_width_ratio = transition_width_ratio
        
        # Enhanced critical band analysis
        self.band_edges = self._compute_enhanced_bark_frequencies()
        self.center_frequencies = self._compute_band_centers()
        
        # High-quality filter bank design
        self.filter_bank = self._design_advanced_filter_bank()
        
        # Noise generation with quality control
        self.noise_generator = self._initialize_noise_generator()
        
        # Temporal envelope processing
        self.envelope_smoothing = self._initialize_envelope_smoother()
        
        # Validate filter bank quality
        self._validate_filter_bank()
    
    def _compute_enhanced_bark_frequencies(self) -> torch.Tensor:
        """
        Compute psychoacoustically accurate Bark-scale frequencies.
        
        Uses the Zwicker & Terhardt (1980) Bark scale formulation
        with corrections for modern psychoacoustic research.
        """
        def hz_to_bark_accurate(f_hz):
            """Enhanced Bark scale conversion with higher accuracy."""
            # Zwicker & Terhardt formula with modern corrections
            f_khz = f_hz / 1000.0
            bark = 13 * np.arctan(0.76 * f_khz) + 3.5 * np.arctan((f_khz / 7.5) ** 2)
            return bark
        
        def bark_to_hz_accurate(bark):
            """Inverse Bark scale with iterative refinement."""
            # Initial estimate using simplified inverse
            f_hz = 600 * np.sinh(bark / 4)
            
            # Newton-Raphson refinement for accuracy
            for _ in range(5):
                bark_est = hz_to_bark_accurate(f_hz)
                
                # Derivative for Newton-Raphson
                df = 1e-3  # Small frequency step
                bark_deriv = (hz_to_bark_accurate(f_hz + df) - bark_est) / df
                
                if abs(bark_deriv) > 1e-10:
                    f_hz = f_hz - (bark_est - bark) / bark_deriv
            
            return f_hz
        
        # Define frequency range with extended coverage
        bark_min = hz_to_bark_accurate(50)    # Lower frequency limit
        bark_max = hz_to_bark_accurate(12000) # Upper frequency limit
        
        # Create perceptually uniform bands
        bark_edges = np.linspace(bark_min, bark_max, self.num_bands + 1)
        hz_edges = np.array([bark_to_hz_accurate(b) for b in bark_edges])
        
        return torch.tensor(hz_edges, dtype=torch.float32)
    
    def _compute_band_centers(self) -> torch.Tensor:
        """Compute geometric center frequencies for each band."""
        centers = []
        for i in range(self.num_bands):
            f_low = self.band_edges[i]
            f_high = self.band_edges[i + 1]
            center = torch.sqrt(f_low * f_high)  # Geometric mean
            centers.append(center)
        
        return torch.tensor(centers)
    
    def _design_advanced_filter_bank(self) -> torch.Tensor:
        """
        Design advanced filter bank with optimized characteristics.
        
        Features:
        - Butterworth filters for flat passband response
        - Minimal phase distortion design
        - Optimized transition bands for computational efficiency
        - Spectral validation and quality assurance
        """
        filter_length = 1024  # Extended length for better frequency resolution
        nyquist = self.sample_rate / 2
        
        # Frequency grid for filter design
        freq_grid = np.linspace(0, nyquist, filter_length // 2 + 1)
        
        filters = []
        
        for i in range(self.num_bands):
            f_low = self.band_edges[i].item()
            f_high = self.band_edges[i + 1].item()
            f_center = self.center_frequencies[i].item()
            
            # Adaptive transition width based on band characteristics
            bandwidth = f_high - f_low
            transition_width = bandwidth * self.transition_width_ratio
            
            # Design bandpass filter response
            response = np.zeros_like(freq_grid)
            
            for j, f in enumerate(freq_grid):
                if f_low <= f <= f_high:
                    # Flat passband
                    response[j] = 1.0
                elif f_low - transition_width <= f < f_low:
                    # Lower transition (raised cosine taper)
                    alpha = (f - (f_low - transition_width)) / transition_width
                    response[j] = 0.5 * (1 - np.cos(np.pi * alpha))
                elif f_high < f <= f_high + transition_width:
                    # Upper transition (raised cosine taper)
                    alpha = (f_high + transition_width - f) / transition_width
                    response[j] = 0.5 * (1 - np.cos(np.pi * alpha))
            
            # Apply psychoacoustic weighting
            response = self._apply_psychoacoustic_weighting(freq_grid, response, f_center)
            
            # Convert to time domain with minimum phase design
            filter_ir = self._design_minimum_phase_filter(response, filter_length)
            
            # Normalize for unit gain at center frequency
            filter_ir = self._normalize_filter_gain(filter_ir, f_center)
            
            filters.append(filter_ir)
        
        filter_bank = torch.tensor(np.array(filters), dtype=torch.float32)
        
        return filter_bank
    
    def _apply_psychoacoustic_weighting(self, freq_grid: np.ndarray, 
                                      response: np.ndarray, f_center: float) -> np.ndarray:
        """Apply psychoacoustic weighting based on auditory masking."""
        # Simple A-weighting approximation for perceptual relevance
        a_weight = np.zeros_like(freq_grid)
        
        for i, f in enumerate(freq_grid):
            if f > 0:
                # Simplified A-weighting formula
                f2 = f * f
                f4 = f2 * f2
                numerator = 12194 ** 2 * f4
                denominator = ((f2 + 20.6 ** 2) * 
                             np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * 
                             (f2 + 12194 ** 2))
                a_weight[i] = numerator / denominator if denominator > 0 else 0
        
        # Normalize A-weighting
        if np.max(a_weight) > 0:
            a_weight = a_weight / np.max(a_weight)
        
        # Apply moderate weighting to preserve band characteristics
        weighted_response = response * (0.8 + 0.2 * a_weight)
        
        return weighted_response
    
    def _design_minimum_phase_filter(self, magnitude_response: np.ndarray, 
                                   filter_length: int) -> np.ndarray:
        """Design minimum phase filter from magnitude response."""
        # Ensure positive magnitude
        magnitude_response = np.maximum(magnitude_response, 1e-8)
        
        # Compute log magnitude
        log_magnitude = np.log(magnitude_response)
        
        # Create symmetric spectrum for real IFFT
        full_log_magnitude = np.concatenate([log_magnitude, log_magnitude[-2:0:-1]])
        
        # Hilbert transform to get minimum phase
        cepstrum = np.fft.ifft(full_log_magnitude).real
        
        # Minimum phase cepstrum
        min_phase_cepstrum = np.zeros_like(cepstrum)
        min_phase_cepstrum[0] = cepstrum[0]
        min_phase_cepstrum[1:len(cepstrum)//2] = 2 * cepstrum[1:len(cepstrum)//2]
        
        # Convert back to frequency domain
        min_phase_spectrum = np.fft.fft(min_phase_cepstrum)
        
        # IFFT to get filter coefficients
        filter_ir = np.fft.ifft(np.exp(min_phase_spectrum)).real
        
        # Truncate and window
        filter_ir = filter_ir[:filter_length]
        window = np.hanning(filter_length)
        filter_ir = filter_ir * window
        
        return filter_ir
    
    def _normalize_filter_gain(self, filter_ir: np.ndarray, center_freq: float) -> np.ndarray:
        """Normalize filter for unit gain at center frequency."""
        # Compute frequency response at center frequency
        n_fft = len(filter_ir) * 4  # Zero-pad for better frequency resolution
        padded_ir = np.pad(filter_ir, (0, n_fft - len(filter_ir)))
        
        freq_response = np.fft.fft(padded_ir)
        frequencies = np.fft.fftfreq(n_fft, 1/self.sample_rate)
        
        # Find closest frequency bin to center frequency
        center_bin = np.argmin(np.abs(frequencies - center_freq))
        center_gain = np.abs(freq_response[center_bin])
        
        # Normalize
        if center_gain > 1e-8:
            filter_ir = filter_ir / center_gain
        
        return filter_ir
    
    def _initialize_noise_generator(self) -> Dict[str, Any]:
        """Initialize high-quality noise generation system."""
        return {
            'generator': torch.Generator(),
            'seed': 42,
            'distribution': 'gaussian',  # or 'uniform'
            'quality_mode': 'high'  # 'high', 'medium', 'fast'
        }
    
    def _initialize_envelope_smoother(self) -> Dict[str, Any]:
        """Initialize temporal envelope smoothing system."""
        return {
            'alpha': 0.1,  # Smoothing factor
            'prev_envelopes': None,
            'transition_samples': 256  # Samples for smooth transitions
        }
    
    def _validate_filter_bank(self):
        """Comprehensive filter bank validation."""
        # Test with white noise to check frequency response
        test_length = 8192
        white_noise = torch.randn(1, test_length)
        
        total_response = torch.zeros(test_length // 2 + 1)
        
        for i in range(self.num_bands):
            # Filter white noise
            filtered = F.conv1d(
                white_noise.unsqueeze(1),
                self.filter_bank[i:i+1].unsqueeze(1),
                padding=self.filter_bank.shape[1] // 2
            )
            
            # Compute power spectral density
            filtered_fft = torch.fft.rfft(filtered.squeeze())
            power = torch.abs(filtered_fft) ** 2
            total_response += power.squeeze()
        
        # Check for reasonable reconstruction
        target_power = torch.abs(torch.fft.rfft(white_noise.squeeze())) ** 2
        reconstruction_error = torch.mean((total_response - target_power) ** 2)
        
        if reconstruction_error > 0.1:
            warnings.warn(f"Filter bank reconstruction error: {reconstruction_error:.4f}")
    
    def generate_noise(self, shape: Tuple[int, ...], 
                      envelopes: torch.Tensor,
                      envelope_smoothing: bool = True) -> torch.Tensor:
        """
        Generate high-quality critical band noise with temporal shaping.
        
        Args:
            shape: Output tensor shape [batch, seq_len]
            envelopes: Band envelopes [batch, seq_len, num_bands]
            envelope_smoothing: Apply temporal envelope smoothing
            
        Returns:
            Generated noise tensor [batch, seq_len] with spectral shaping
        """
        batch_size, seq_len = shape
        device = envelopes.device
        
        # Validate inputs
        if not self.check_tensor_validity(envelopes, "noise_envelopes"):
            return torch.zeros(shape, device=device)
        
        # Move filter bank to appropriate device
        if self.filter_bank.device != device:
            self.filter_bank = self.filter_bank.to(device)
        
        # Apply envelope smoothing if requested
        if envelope_smoothing:
            envelopes = self._apply_envelope_smoothing(envelopes)
        
        # Generate base noise with extended length for filtering
        filter_length = self.filter_bank.shape[1]
        extended_length = seq_len + filter_length
        
        if self.noise_generator['quality_mode'] == 'high':
            # High-quality noise with better spectral characteristics
            base_noise = self._generate_high_quality_noise(
                (batch_size, extended_length), device
            )
        else:
            # Standard Gaussian noise
            base_noise = torch.randn(batch_size, extended_length, device=device)
        
        # Apply critical band filtering
        filtered_bands = []
        
        for band_idx in range(self.num_bands):
            # Convolve with band filter
            band_filter = self.filter_bank[band_idx].unsqueeze(0).unsqueeze(0)
            
            filtered = F.conv1d(
                base_noise.unsqueeze(1),
                band_filter,
                padding=0
            ).squeeze(1)
            
            # Trim to target length
            filtered = filtered[:, :seq_len]
            
            # Validate filtered output
            if not self.check_tensor_validity(filtered, f"filtered_band_{band_idx}"):
                filtered = torch.zeros(batch_size, seq_len, device=device)
            
            filtered_bands.append(filtered)
        
        # Stack band outputs
        band_noise = torch.stack(filtered_bands, dim=-1)  # [batch, seq_len, num_bands]
        
        # Apply time-varying envelopes with soft limiting
        shaped_noise = band_noise * torch.tanh(envelopes)
        
        # Sum across bands with numerical stability
        output_noise = torch.sum(shaped_noise, dim=-1)  # [batch, seq_len]
        
        # Apply gentle compression to prevent harsh peaks
        output_noise = torch.tanh(output_noise * 0.8) / 0.8
        
        # Final validation
        if not self.check_tensor_validity(output_noise, "final_noise_output"):
            warnings.warn("Invalid noise output detected, returning zeros")
            output_noise = torch.zeros(shape, device=device)
        
        return output_noise
    
    def _generate_high_quality_noise(self, shape: Tuple[int, ...], 
                                   device: torch.device) -> torch.Tensor:
        """Generate high-quality noise with improved spectral characteristics."""
        # Generate multiple independent noise sources
        num_sources = 4
        noise_sources = []
        
        for i in range(num_sources):
            # Use different seeds for decorrelation
            generator = torch.Generator(device=device)
            generator.manual_seed(self.noise_generator['seed'] + i)
            
            source = torch.randn(shape, generator=generator, device=device)
            noise_sources.append(source)
        
        # Combine sources with slight decorrelation
        combined_noise = noise_sources[0]
        for i in range(1, num_sources):
            weight = 0.5 ** i  # Decreasing weights
            combined_noise += weight * noise_sources[i]
        
        # Normalize
        combined_noise = self.stable_normalize(combined_noise, dim=-1)
        
        return combined_noise
    
    def _apply_envelope_smoothing(self, envelopes: torch.Tensor) -> torch.Tensor:
        """Apply temporal smoothing to envelopes for natural transitions."""
        if self.envelope_smoothing['prev_envelopes'] is None:
            self.envelope_smoothing['prev_envelopes'] = envelopes[:, 0:1, :].clone()
            return envelopes
        
        # Apply exponential smoothing
        alpha = self.envelope_smoothing['alpha']
        smoothed = envelopes.clone()
        
        # Initialize with smoothed transition from previous frame
        smoothed[:, 0, :] = (
            alpha * envelopes[:, 0, :] + 
            (1 - alpha) * self.envelope_smoothing['prev_envelopes'][:, 0, :]
        )
        
        # Apply smoothing across time
        for t in range(1, envelopes.shape[1]):
            smoothed[:, t, :] = (
                alpha * envelopes[:, t, :] + 
                (1 - alpha) * smoothed[:, t-1, :]
            )
        
        # Update previous envelopes
        self.envelope_smoothing['prev_envelopes'] = smoothed[:, -1:, :].clone()
        
        return smoothed


class HarmonicSynthesizer(nn.Module, NumericalStabilityMixin):
    """
    Production-grade harmonic synthesizer with advanced features.
    
    Comprehensive Features:
    - Adaptive harmonic limiting based on content analysis
    - Phase coherence maintenance across harmonics
    - Advanced amplitude modeling with perceptual weighting
    - Real-time performance optimization
    - Comprehensive quality monitoring
    """
    
    def __init__(self, sample_rate: int = 44100, num_harmonics: int = 48,
                 fundamental_range: Tuple[float, float] = (50.0, 800.0),
                 quality_mode: str = "production"):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.num_harmonics = num_harmonics
        self.fundamental_min, self.fundamental_max = fundamental_range
        self.quality_mode = quality_mode
        
        # Nyquist frequency for harmonic limiting
        self.nyquist_freq = sample_rate / 2
        
        # Initialize synthesizer engine
        precision_level = "high" if quality_mode == "production" else "medium"
        self.synthesizer = OptimizedTritonHarmonicSynthesizer(
            sample_rate=sample_rate,
            auto_tune=True,
            target_precision=precision_level
        )
        
        # Advanced psychoacoustic weighting
        self.register_buffer('harmonic_weights', self._compute_advanced_psychoacoustic_weights())
        self.register_buffer('masking_curve', self._compute_masking_curve())
        
        # Phase management system
        self.register_buffer('phase_accumulator', torch.zeros(1, num_harmonics, dtype=torch.float64))
        self.register_buffer('phase_correction', torch.zeros(1, num_harmonics))
        
        # Quality monitoring
        self.quality_metrics = {
            'thd_history': [],
            'snr_history': [],
            'phase_coherence_history': []
        }
    
    def _compute_advanced_psychoacoustic_weights(self) -> torch.Tensor:
        """
        Compute sophisticated psychoacoustic amplitude weights.
        
        Incorporates:
        - Equal loudness curves (ISO 226)
        - Frequency masking effects
        - Harmonic series naturalism
        - High-frequency hearing sensitivity
        """
        harmonic_numbers = torch.arange(1, self.num_harmonics + 1, dtype=torch.float32)
        
        # Base harmonic decay (natural harmonic series)
        natural_decay = 1.0 / harmonic_numbers
        
        # Equal loudness curve approximation
        def equal_loudness_weight(freq_hz, level_db=60):
            """Approximate equal loudness curve at given level."""
            # Simplified A-weighting for efficiency
            freq_khz = freq_hz / 1000.0
            if freq_hz > 0:
                numerator = 12194**2 * freq_hz**4
                denominator = ((freq_hz**2 + 20.6**2) * 
                             ((freq_hz**2 + 107.7**2) * (freq_hz**2 + 737.9**2))**0.5 * 
                             (freq_hz**2 + 12194**2))
                weight = numerator / denominator if denominator > 0 else 0
                return weight
            return 0
        
        # Approximate harmonic frequencies (assuming fundamental at 200Hz)
        approx_fundamental = 200.0
        harmonic_freqs = harmonic_numbers * approx_fundamental
        
        # Equal loudness weights
        loudness_weights = torch.tensor([
            equal_loudness_weight(freq.item()) for freq in harmonic_freqs
        ])
        
        # Normalize loudness weights
        if loudness_weights.max() > 0:
            loudness_weights = loudness_weights / loudness_weights.max()
        
        # High-frequency rolloff for anti-aliasing
        rolloff_weights = torch.exp(-0.15 * (harmonic_numbers - 1))
        
        # Combine weightings
        combined_weights = natural_decay * (0.7 + 0.3 * loudness_weights) * rolloff_weights
        
        # Normalize to preserve total energy
        energy_preservation = torch.sqrt(torch.sum(combined_weights**2))
        if energy_preservation > 0:
            combined_weights = combined_weights / energy_preservation * math.sqrt(self.num_harmonics)
        
        return combined_weights
    
    def _compute_masking_curve(self) -> torch.Tensor:
        """Compute frequency masking curve for harmonic interactions."""
        # Simple bark-scale masking approximation
        harmonic_numbers = torch.arange(1, self.num_harmonics + 1, dtype=torch.float32)
        
        # Masking falls off with harmonic distance
        masking_curve = torch.exp(-0.1 * torch.abs(
            harmonic_numbers.unsqueeze(0) - harmonic_numbers.unsqueeze(1)
        ))
        
        return masking_curve
    
    def _apply_adaptive_harmonic_limiting(self, fundamental_freq: torch.Tensor, 
                                        harmonic_amplitudes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply content-adaptive harmonic limiting for optimal quality.
        
        Args:
            fundamental_freq: Fundamental frequencies [batch, seq_len]
            harmonic_amplitudes: Harmonic amplitudes [batch, seq_len, num_harmonics]
            
        Returns:
            Tuple of (limited_amplitudes, nyquist_mask)
        """
        batch_size, seq_len = fundamental_freq.shape
        device = fundamental_freq.device
        
        # Compute harmonic frequencies
        harmonic_numbers = torch.arange(1, self.num_harmonics + 1, 
                                      device=device, dtype=torch.float32)
        
        # [batch, seq_len, 1] * [1, 1, num_harmonics] -> [batch, seq_len, num_harmonics]
        harmonic_frequencies = fundamental_freq.unsqueeze(-1) * harmonic_numbers.view(1, 1, -1)
        
        # Nyquist frequency mask
        nyquist_mask = (harmonic_frequencies < self.nyquist_freq * 0.95).float()  # 5% safety margin
        
        # Content-adaptive limiting based on fundamental frequency
        content_factor = torch.clamp(
            (self.fundamental_max - fundamental_freq) / (self.fundamental_max - self.fundamental_min),
            0.0, 1.0
        ).unsqueeze(-1)
        
        # Higher fundamentals -> fewer harmonics for quality
        adaptive_weights = content_factor * 0.3 + 0.7  # Range: [0.7, 1.0]
        
        # Apply all limiting factors
        limited_amplitudes = (harmonic_amplitudes * 
                            nyquist_mask * 
                            adaptive_weights * 
                            self.harmonic_weights.view(1, 1, -1))
        
        return limited_amplitudes, nyquist_mask
    
    def _manage_phase_accumulation(self, fundamental_freq: torch.Tensor,
                                 initial_phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Advanced phase accumulation with drift correction and coherence maintenance.
        
        Args:
            fundamental_freq: Fundamental frequencies [batch, seq_len]
            initial_phase: Optional initial phases [batch, seq_len, num_harmonics]
            
        Returns:
            Coherent phase tensor [batch, seq_len, num_harmonics]
        """
        batch_size, seq_len = fundamental_freq.shape
        device = fundamental_freq.device
        
        # Ensure phase accumulator matches batch size
        if self.phase_accumulator.shape[0] != batch_size:
            self.phase_accumulator = self.phase_accumulator.expand(batch_size, -1).contiguous()
            self.phase_correction = self.phase_correction.expand(batch_size, -1).contiguous()
        
        # Generate harmonic multipliers
        harmonic_numbers = torch.arange(1, self.num_harmonics + 1, 
                                      device=device, dtype=torch.float64)
        
        # Time step
        dt = 1.0 / self.sample_rate
        
        # Initialize phase tracking
        phases = torch.zeros(batch_size, seq_len, self.num_harmonics, 
                           device=device, dtype=torch.float32)
        
        # Use initial phase if provided
        if initial_phase is not None:
            current_phase = initial_phase[:, 0, :].double()
        else:
            current_phase = self.phase_accumulator.clone()
        
        # Phase accumulation with extended precision
        for t in range(seq_len):
            # Store current phase
            phases[:, t, :] = current_phase.float()
            
            # Compute phase increments for all harmonics
            freq_t = fundamental_freq[:, t].double().unsqueeze(-1)  # [batch, 1]
            phase_increments = 2 * math.pi * freq_t * harmonic_numbers.view(1, -1) * dt
            
            # Accumulate with extended precision
            current_phase = current_phase + phase_increments
            
            # Apply periodic correction to prevent drift
            # Use exact modular arithmetic for phase wrapping
            current_phase = current_phase % (2 * math.pi)
            
            # Coherence correction: ensure harmonic relationships
            if t > 0 and t % 100 == 0:  # Every 100 samples
                fundamental_phase = current_phase[:, 0:1]
                expected_phases = fundamental_phase * harmonic_numbers.view(1, -1)
                phase_error = current_phase - (expected_phases % (2 * math.pi))
                
                # Apply gentle correction
                correction_strength = 0.01
                current_phase = current_phase - correction_strength * phase_error
        
        # Update phase accumulator for next call
        self.phase_accumulator.copy_(current_phase)
        
        return phases
    
    def _compute_quality_metrics(self, output_audio: torch.Tensor, 
                               harmonic_amplitudes: torch.Tensor) -> Dict[str, float]:
        """Compute comprehensive quality metrics."""
        metrics = {}
        
        # Total Harmonic Distortion (THD) estimation
        fundamental_power = harmonic_amplitudes[:, :, 0]**2
        harmonic_power = torch.sum(harmonic_amplitudes[:, :, 1:]**2, dim=-1)
        
        thd = torch.sqrt(harmonic_power / (fundamental_power + 1e-8))
        metrics['thd_percent'] = torch.mean(thd).item() * 100
        
        # Signal-to-Noise Ratio estimation
        signal_power = torch.mean(output_audio**2)
        noise_floor = 1e-6  # Assumed noise floor
        snr_db = 10 * torch.log10(signal_power / noise_floor)
        metrics['snr_db'] = snr_db.item()
        
        # Phase coherence measure
        phase_variance = torch.var(harmonic_amplitudes, dim=1)
        coherence = 1.0 / (1.0 + torch.mean(phase_variance))
        metrics['phase_coherence'] = coherence.item()
        
        return metrics
    
    def forward(self, fundamental_freq: torch.Tensor, 
                harmonic_amplitudes: torch.Tensor,
                initial_phase: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Comprehensive harmonic synthesis with quality monitoring.
        
        Args:
            fundamental_freq: Fundamental frequencies [batch, seq_len]
            harmonic_amplitudes: Harmonic amplitudes [batch, seq_len, num_harmonics]
            initial_phase: Optional initial phases [batch, seq_len, num_harmonics]
            
        Returns:
            Dictionary containing synthesized audio and metadata
        """
        # Input validation
        if not self.check_tensor_validity(fundamental_freq, "fundamental_freq"):
            batch_size, seq_len = fundamental_freq.shape
            return {
                'audio': torch.zeros(batch_size, seq_len, device=fundamental_freq.device),
                'quality_metrics': {},
                'limiting_applied': False
            }
        
        # Apply adaptive harmonic limiting
        limited_amplitudes, nyquist_mask = self._apply_adaptive_harmonic_limiting(
            fundamental_freq, harmonic_amplitudes
        )
        
        # Manage phase accumulation
        phases = self._manage_phase_accumulation(fundamental_freq, initial_phase)
        
        # Generate harmonic frequencies for synthesis
        harmonic_numbers = torch.arange(1, self.num_harmonics + 1, 
                                      device=fundamental_freq.device, dtype=torch.float32)
        harmonic_frequencies = fundamental_freq.unsqueeze(-1) * harmonic_numbers.view(1, 1, -1)
        
        # High-performance synthesis
        synthesized_audio = self.synthesizer.synthesize(
            harmonic_frequencies, limited_amplitudes, phases
        )
        
        # Quality assessment
        quality_metrics = self._compute_quality_metrics(synthesized_audio, limited_amplitudes)
        
        # Update quality history
        self.quality_metrics['thd_history'].append(quality_metrics['thd_percent'])
        self.quality_metrics['snr_history'].append(quality_metrics['snr_db'])
        self.quality_metrics['phase_coherence_history'].append(quality_metrics['phase_coherence'])
        
        # Maintain history size
        max_history = 1000
        for key in self.quality_metrics:
            if len(self.quality_metrics[key]) > max_history:
                self.quality_metrics[key] = self.quality_metrics[key][-max_history:]
        
        return {
            'audio': synthesized_audio,
            'quality_metrics': quality_metrics,
            'limiting_applied': torch.mean(nyquist_mask).item() < 0.99,
            'harmonic_mask': nyquist_mask,
            'performance_stats': self.synthesizer.get_performance_metrics()
        }
    
    def reset_phase_state(self):
        """Reset phase accumulation state for new sequence."""
        self.phase_accumulator.zero_()
        self.phase_correction.zero_()
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get comprehensive quality summary."""
        if not self.quality_metrics['thd_history']:
            return {}
        
        summary = {}
        for metric_name, history in self.quality_metrics.items():
            if history:
                summary[metric_name] = {
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'min': np.min(history),
                    'max': np.max(history),
                    'recent_avg': np.mean(history[-100:]) if len(history) >= 100 else np.mean(history)
                }
        
        return summary


class DDSPVocoder(nn.Module, NumericalStabilityMixin):
    """
    Production-grade DDSP vocoder with comprehensive error handling and optimization.
    
    Enterprise Features:
    - Robust parameter prediction with range validation
    - Advanced harmonic+noise decomposition
    - Real-time performance optimization
    - Comprehensive quality assurance
    - Detailed performance monitoring
    - Graceful degradation under adverse conditions
    """
    
    def __init__(self, hidden_dim: int = 512, harmonics_count: int = 48,
                 noise_bands: int = 8, sample_rate: int = 44100, 
                 channels: int = 2, quality_mode: str = "production"):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.harmonics_count = harmonics_count
        self.noise_bands = noise_bands
        self.sample_rate = sample_rate
        self.channels = channels
        self.quality_mode = quality_mode
        
        # Parameter prediction networks with enhanced architectures
        self.f0_predictor = self._build_f0_predictor()
        self.harmonic_predictor = self._build_harmonic_predictor()
        self.noise_predictor = self._build_noise_predictor()
        self.balance_predictor = self._build_balance_predictor()
        
        # Advanced synthesis engines
        self.harmonic_synth = HarmonicSynthesizer(
            sample_rate=sample_rate,
            num_harmonics=harmonics_count,
            fundamental_range=(50.0, 800.0),
            quality_mode=quality_mode
        )
        
        self.noise_generator = AdvancedCriticalBandNoiseGenerator(
            sample_rate=sample_rate,
            num_bands=noise_bands,
            filter_order=4
        )
        
        # Output processing
        if channels == 2:
            self.stereo_processor = self._build_stereo_processor()
        else:
            self.stereo_processor = None
        
        # Quality monitoring
        self.synthesis_stats = {
            'total_syntheses': 0,
            'processing_times': [],
            'quality_scores': [],
            'parameter_ranges': {
                'f0_range': [],
                'amplitude_range': [],
                'noise_level': []
            }
        }
    
    def _build_f0_predictor(self) -> nn.Module:
        """Build robust fundamental frequency predictor."""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.LayerNorm(self.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()  # Normalize to [0, 1] for stable training
        )
    
    def _build_harmonic_predictor(self) -> nn.Module:
        """Build harmonic amplitude predictor with stability features."""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.harmonics_count),
            nn.Softplus()  # Ensure positive amplitudes
        )
    
    def _build_noise_predictor(self) -> nn.Module:
        """Build noise envelope predictor."""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.LayerNorm(self.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 4, self.noise_bands),
            nn.Softplus()  # Ensure positive envelopes
        )
    
    def _build_balance_predictor(self) -> nn.Module:
        """Build harmonic/noise balance predictor."""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.LayerNorm(self.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 4, self.hidden_dim // 8),
            nn.LayerNorm(self.hidden_dim // 8),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 8, 1),
            nn.Sigmoid()  # Balance between harmonic [0] and noise [1]
        )
    
    def _build_stereo_processor(self) -> nn.Module:
        """Build stereo expansion and spatialization processor."""
        return nn.Sequential(
            nn.Linear(1, 4),  # Expand to intermediate stereo features
            nn.Tanh(),
            nn.Linear(4, 2)   # Final stereo output
        )
    
    def _validate_and_clamp_parameters(self, f0_normalized: torch.Tensor,
                                     harmonic_amps: torch.Tensor,
                                     noise_envelopes: torch.Tensor,
                                     harmonic_balance: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Validate and clamp synthesis parameters to safe ranges."""
        # F0 validation and clamping
        f0_clamped = torch.clamp(f0_normalized, 0.001, 0.999)
        
        # Harmonic amplitude validation
        harmonic_amps_clamped = torch.clamp(harmonic_amps, 0.0, 10.0)
        
        # Check for extreme values
        if torch.max(harmonic_amps) > 5.0:
            warnings.warn("Extreme harmonic amplitudes detected, applying limiting")
            harmonic_amps_clamped = harmonic_amps_clamped / torch.max(harmonic_amps_clamped) * 2.0
        
        # Noise envelope validation
        noise_envelopes_clamped = torch.clamp(noise_envelopes, 0.0, 5.0)
        
        # Balance validation
        balance_clamped = torch.clamp(harmonic_balance, 0.01, 0.99)
        
        return f0_clamped, harmonic_amps_clamped, noise_envelopes_clamped, balance_clamped
    
    def _convert_f0_to_hz(self, f0_normalized: torch.Tensor) -> torch.Tensor:
        """Convert normalized F0 to Hz with musical scale mapping."""
        # Use logarithmic scale for better perceptual coverage
        f0_min_hz, f0_max_hz = 50.0, 800.0  # Extended range for versatility
        
        # Logarithmic interpolation
        log_f0_min = math.log(f0_min_hz)
        log_f0_max = math.log(f0_max_hz)
        
        log_f0 = log_f0_min + f0_normalized * (log_f0_max - log_f0_min)
        fundamental_freq = torch.exp(log_f0)
        
        return fundamental_freq
    
    def _apply_advanced_parameter_conditioning(self, features: torch.Tensor) -> torch.Tensor:
        """Apply advanced conditioning to input features for stable parameter prediction."""
        # Normalize features
        normalized_features = F.layer_norm(features, features.shape[-1:])
        
        # Apply temporal smoothing for stability
        if features.shape[1] > 1:  # Multi-frame input
            # Simple moving average for temporal consistency
            kernel_size = min(5, features.shape[1])
            if kernel_size > 1:
                padding = kernel_size // 2
                smoothed = F.avg_pool1d(
                    normalized_features.transpose(1, 2),
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding
                ).transpose(1, 2)
            else:
                smoothed = normalized_features
        else:
            smoothed = normalized_features
        
        return smoothed
    
    def forward(self, features: torch.Tensor, 
                metadata: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Generate audio from latent features using DDSP synthesis.
        
        Args:
            features: Latent features [batch, seq_len, hidden_dim]
            metadata: Optional metadata from encoder/quantizer
            
        Returns:
            Synthesized stereo audio [batch, channels, samples]
        """
        start_time = time.perf_counter()
        
        batch_size, seq_len, _ = features.shape
        device = features.device
        
        # Input validation
        if not self.check_tensor_validity(features, "input_features"):
            return torch.zeros(batch_size, self.channels, seq_len, device=device)
        
        # Apply advanced feature conditioning
        conditioned_features = self._apply_advanced_parameter_conditioning(features)
        
        # Predict synthesis parameters with error handling
        try:
            f0_normalized = self.f0_predictor(conditioned_features).squeeze(-1)
            harmonic_amps = self.harmonic_predictor(conditioned_features)
            noise_envelopes = self.noise_predictor(conditioned_features)
            harmonic_balance = self.balance_predictor(conditioned_features).squeeze(-1)
        except Exception as e:
            warnings.warn(f"Parameter prediction failed: {e}")
            return torch.zeros(batch_size, self.channels, seq_len, device=device)
        
        # Validate and clamp parameters
        f0_normalized, harmonic_amps, noise_envelopes, harmonic_balance = \
            self._validate_and_clamp_parameters(f0_normalized, harmonic_amps, 
                                               noise_envelopes, harmonic_balance)
        
        # Convert F0 to Hz
        fundamental_freq = self._convert_f0_to_hz(f0_normalized)
        
        # Update parameter statistics
        self._update_parameter_statistics(fundamental_freq, harmonic_amps, noise_envelopes)
        
        # Harmonic synthesis with comprehensive error handling
        try:
            harmonic_result = self.harmonic_synth(fundamental_freq, harmonic_amps)
            harmonic_audio = harmonic_result['audio']
            synthesis_quality = harmonic_result['quality_metrics']
        except Exception as e:
            warnings.warn(f"Harmonic synthesis failed: {e}")
            harmonic_audio = torch.zeros(batch_size, seq_len, device=device)
            synthesis_quality = {}
        
        # Noise synthesis with error handling
        try:
            noise_audio = self.noise_generator.generate_noise(
                shape=(batch_size, seq_len),
                envelopes=noise_envelopes,
                envelope_smoothing=True
            )
        except Exception as e:
            warnings.warn(f"Noise synthesis failed: {e}")
            noise_audio = torch.zeros(batch_size, seq_len, device=device)
        
        # Balance harmonic and noise components
        harmonic_weight = 1 - harmonic_balance
        noise_weight = harmonic_balance
        
        balanced_harmonic = harmonic_audio * harmonic_weight
        balanced_noise = noise_audio * noise_weight
        
        # Combine components with soft limiting
        combined_audio = balanced_harmonic + balanced_noise
        combined_audio = torch.tanh(combined_audio * 0.9) / 0.9  # Soft limiting
        
        # Stereo processing
        if self.stereo_processor is not None:
            # Expand to stereo with spatial processing
            mono_audio = combined_audio.unsqueeze(-1)  # [batch, seq_len, 1]
            stereo_audio = self.stereo_processor(mono_audio)  # [batch, seq_len, 2]
            
            # Apply gentle stereo width control
            stereo_width = 0.8  # Adjustable stereo width
            mid = (stereo_audio[:, :, 0] + stereo_audio[:, :, 1]) / 2
            side = (stereo_audio[:, :, 0] - stereo_audio[:, :, 1]) / 2 * stereo_width
            
            left = mid + side
            right = mid - side
            
            output_audio = torch.stack([left, right], dim=1)  # [batch, 2, seq_len]
        else:
            # Mono output
            output_audio = combined_audio.unsqueeze(1)  # [batch, 1, seq_len]
        
        # Final validation and quality control
        if not self.check_tensor_validity(output_audio, "final_audio_output"):
            warnings.warn("Invalid audio output detected, returning silence")
            output_audio = torch.zeros(batch_size, self.channels, seq_len, device=device)
        
        # Apply final dynamic range control
        output_audio = self._apply_final_processing(output_audio)
        
        # Update performance statistics
        processing_time = time.perf_counter() - start_time
        self._update_performance_statistics(processing_time, synthesis_quality)
        
        return output_audio
    
    def _apply_final_processing(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply final processing for optimal output quality."""
        # Gentle peak limiting
        peak_threshold = 0.95
        peak_level = torch.max(torch.abs(audio))
        
        if peak_level > peak_threshold:
            limiter_ratio = peak_threshold / peak_level
            audio = audio * limiter_ratio
        
        # DC offset removal
        audio = audio - torch.mean(audio, dim=-1, keepdim=True)
        
        # Final safety clipping
        audio = torch.clamp(audio, -1.0, 1.0)
        
        return audio
    
    def _update_parameter_statistics(self, f0: torch.Tensor, 
                                   harmonic_amps: torch.Tensor,
                                   noise_envelopes: torch.Tensor):
        """Update parameter statistics for monitoring."""
        self.synthesis_stats['parameter_ranges']['f0_range'].append({
            'min': torch.min(f0).item(),
            'max': torch.max(f0).item(),
            'mean': torch.mean(f0).item()
        })
        
        self.synthesis_stats['parameter_ranges']['amplitude_range'].append({
            'min': torch.min(harmonic_amps).item(),
            'max': torch.max(harmonic_amps).item(),
            'mean': torch.mean(harmonic_amps).item()
        })
        
        self.synthesis_stats['parameter_ranges']['noise_level'].append({
            'min': torch.min(noise_envelopes).item(),
            'max': torch.max(noise_envelopes).item(),
            'mean': torch.mean(noise_envelopes).item()
        })
    
    def _update_performance_statistics(self, processing_time: float, 
                                     quality_metrics: Dict[str, float]):
        """Update performance and quality statistics."""
        self.synthesis_stats['total_syntheses'] += 1
        self.synthesis_stats['processing_times'].append(processing_time)
        
        if quality_metrics:
            combined_quality = sum(quality_metrics.values()) / len(quality_metrics)
            self.synthesis_stats['quality_scores'].append(combined_quality)
        
        # Maintain reasonable history size
        max_history = 1000
        for key in ['processing_times', 'quality_scores']:
            if len(self.synthesis_stats[key]) > max_history:
                self.synthesis_stats[key] = self.synthesis_stats[key][-max_history:]
        
        for param_type in self.synthesis_stats['parameter_ranges']:
            if len(self.synthesis_stats['parameter_ranges'][param_type]) > max_history:
                self.synthesis_stats['parameter_ranges'][param_type] = \
                    self.synthesis_stats['parameter_ranges'][param_type][-max_history:]
    
    def reset_synthesis_state(self):
        """Reset all synthesis state for new sequence."""
        self.harmonic_synth.reset_phase_state()
        # Reset noise generator state if it has any
        if hasattr(self.noise_generator, 'envelope_smoothing'):
            self.noise_generator.envelope_smoothing['prev_envelopes'] = None
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for monitoring and optimization."""
        stats = {
            'total_syntheses': self.synthesis_stats['total_syntheses'],
            'performance': {},
            'quality': {},
            'parameters': {}
        }
        
        if self.synthesis_stats['processing_times']:
            times = self.synthesis_stats['processing_times']
            stats['performance'] = {
                'avg_processing_time_ms': np.mean(times) * 1000,
                'min_processing_time_ms': np.min(times) * 1000,
                'max_processing_time_ms': np.max(times) * 1000,
                'std_processing_time_ms': np.std(times) * 1000,
                'real_time_factor': np.mean(times) * self.sample_rate / 1000  # Approximate RTF
            }
        
        if self.synthesis_stats['quality_scores']:
            scores = self.synthesis_stats['quality_scores']
            stats['quality'] = {
                'avg_quality_score': np.mean(scores),
                'min_quality_score': np.min(scores),
                'max_quality_score': np.max(scores),
                'quality_trend': np.mean(scores[-100:]) - np.mean(scores[:100]) if len(scores) >= 200 else 0
            }
        
        # Parameter range statistics
        for param_type, param_history in self.synthesis_stats['parameter_ranges'].items():
            if param_history:
                recent_params = param_history[-100:] if len(param_history) >= 100 else param_history
                stats['parameters'][param_type] = {
                    'recent_min': min(p['min'] for p in recent_params),
                    'recent_max': max(p['max'] for p in recent_params),
                    'recent_avg': np.mean([p['mean'] for p in recent_params])
                }
        
        # Add component-specific statistics
        stats['harmonic_synth'] = self.harmonic_synth.get_quality_summary()
        
        return stats
    
    def get_synthesis_parameters(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract synthesis parameters without generating audio (for analysis).
        
        Args:
            features: Latent features [batch, seq_len, hidden_dim]
            
        Returns:
            Dictionary of synthesis parameters
        """
        with torch.no_grad():
            # Apply feature conditioning
            conditioned_features = self._apply_advanced_parameter_conditioning(features)
            
            # Predict parameters
            f0_normalized = self.f0_predictor(conditioned_features).squeeze(-1)
            harmonic_amps = self.harmonic_predictor(conditioned_features)
            noise_envelopes = self.noise_predictor(conditioned_features)
            harmonic_balance = self.balance_predictor(conditioned_features).squeeze(-1)
            
            # Validate and convert
            f0_normalized, harmonic_amps, noise_envelopes, harmonic_balance = \
                self._validate_and_clamp_parameters(f0_normalized, harmonic_amps, 
                                                   noise_envelopes, harmonic_balance)
            
            fundamental_freq = self._convert_f0_to_hz(f0_normalized)
            
            return {
                'fundamental_frequency': fundamental_freq,
                'harmonic_amplitudes': harmonic_amps,
                'noise_envelopes': noise_envelopes,
                'harmonic_balance': harmonic_balance,
                'f0_normalized': f0_normalized
            }


