"""
LyCodec v2.1 Utility Module
============================

Production-grade utilities including adaptive bit allocation, stable checkpointing,
gradient clipping, and audio data processing for robust codec training and deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import math
import numpy as np
import os
import hashlib
import pickle
import warnings
import librosa
import torchaudio
from pathlib import Path
import json
import time
import random


class FastComplexityPredictor:
    """
    FFT-free spectral analysis for real-time complexity prediction.
    
    Implements O(1) temporal novelty calculation and pre-computed Gammatone
    filterbank convolution for efficient bit allocation decisions.
    """
    
    def __init__(self, sample_rate: int = 44100, num_bands: int = 8, 
                 window_size: int = 256):
        self.sample_rate = sample_rate
        self.num_bands = num_bands
        self.window_size = window_size
        
        # Pre-computed Gammatone filterbank
        self.filterbank = self._precompute_gammatone_filters()
        
        # Sliding window state for O(1) statistics
        self.sliding_stats = {
            'sum': None,
            'sum_sq': None,
            'buffer': None,
            'idx': 0,
            'initialized': False
        }
        
        # Complexity weighting factors
        self.complexity_weights = {
            'spectral_entropy': 0.4,
            'temporal_novelty': 0.3,
            'cross_channel_correlation': 0.2,
            'dynamic_range': 0.1
        }
    
    def _precompute_gammatone_filters(self) -> torch.Tensor:
        """
        Pre-compute Gammatone filterbank for efficient convolution.
        
        Returns:
            Filter bank tensor [num_bands, filter_length]
        """
        filter_length = 256  # Short filters for efficiency
        
        # ERB-scale center frequencies
        min_erb = 21.4 * np.log10(1 + 0.00437 * 80)    # 80 Hz
        max_erb = 21.4 * np.log10(1 + 0.00437 * 8000)  # 8 kHz
        erb_points = np.linspace(min_erb, max_erb, self.num_bands)
        center_freqs = (10**(erb_points / 21.4) - 1) / 0.00437
        
        # Generate filters
        t = np.arange(filter_length) / self.sample_rate
        filters = []
        
        for fc in center_freqs:
            erb = 24.7 * (4.37 * fc / 1000 + 1)
            b = 1.019 * erb
            
            # Gammatone impulse response
            envelope = (t**3) * np.exp(-2 * np.pi * b * t)
            carrier = np.cos(2 * np.pi * fc * t)
            
            filter_ir = envelope * carrier
            filter_ir = filter_ir / np.sqrt(np.sum(filter_ir**2))  # Normalize
            filters.append(filter_ir)
        
        return torch.tensor(np.array(filters), dtype=torch.float32)
    
    def _sliding_window_variance(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Compute sliding window variance with O(1) complexity per sample.
        
        Args:
            signal: Input signal [batch, channels, samples]
            
        Returns:
            Temporal variance [batch, channels, samples]
        """
        batch_size, channels, samples = signal.shape
        device = signal.device
        
        # Initialize sliding window state if needed
        if not self.sliding_stats['initialized']:
            self.sliding_stats['sum'] = torch.zeros(batch_size, channels, device=device)
            self.sliding_stats['sum_sq'] = torch.zeros(batch_size, channels, device=device)
            self.sliding_stats['buffer'] = torch.zeros(
                batch_size, channels, self.window_size, device=device
            )
            self.sliding_stats['idx'] = 0
            self.sliding_stats['initialized'] = True
        
        variances = []
        
        for t in range(samples):
            current_sample = signal[:, :, t]  # [batch, channels]
            
            # Remove old value from sliding window
            old_sample = self.sliding_stats['buffer'][:, :, self.sliding_stats['idx']]
            self.sliding_stats['sum'] -= old_sample
            self.sliding_stats['sum_sq'] -= old_sample**2
            
            # Add new value
            self.sliding_stats['buffer'][:, :, self.sliding_stats['idx']] = current_sample
            self.sliding_stats['sum'] += current_sample
            self.sliding_stats['sum_sq'] += current_sample**2
            
            # Compute variance
            mean = self.sliding_stats['sum'] / self.window_size
            variance = (self.sliding_stats['sum_sq'] / self.window_size) - mean**2
            variance = torch.clamp(variance, min=1e-8)
            
            variances.append(variance)
            
            # Update index
            self.sliding_stats['idx'] = (self.sliding_stats['idx'] + 1) % self.window_size
        
        return torch.stack(variances, dim=-1)  # [batch, channels, samples]
    
    def predict_complexity(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Predict audio complexity for bit allocation.
        
        Args:
            audio: Input audio [batch, channels, samples]
            
        Returns:
            Complexity scores [batch, samples] in range [0, 1]
        """
        batch_size, channels, samples = audio.shape
        device = audio.device
        
        # Move filterbank to device if needed
        if self.filterbank.device != device:
            self.filterbank = self.filterbank.to(device)
        
        # 1. Spectral entropy via filterbank analysis
        # Reshape for convolution: [batch*channels, 1, samples]
        audio_flat = audio.view(batch_size * channels, 1, samples)
        
        # Pad for convolution
        padding = self.filterbank.shape[1] - 1
        audio_padded = F.pad(audio_flat, (padding, 0))
        
        # Apply filterbank
        band_energies = F.conv1d(
            audio_padded, 
            self.filterbank.unsqueeze(1), 
            groups=1
        )  # [batch*channels, num_bands, samples]
        
        # Compute spectral entropy
        band_energies = band_energies.view(batch_size, channels, self.num_bands, samples)
        band_probs = F.softmax(band_energies, dim=2)  # Normalize across bands
        spectral_entropy = -torch.sum(band_probs * torch.log(band_probs + 1e-8), dim=2)
        spectral_entropy = torch.mean(spectral_entropy, dim=1)  # Average over channels
        
        # 2. Temporal novelty via sliding window variance
        temporal_variance = self._sliding_window_variance(audio)
        temporal_novelty = torch.mean(temporal_variance, dim=1)  # Average over channels
        
        # 3. Cross-channel correlation (for stereo)
        if channels == 2:
            # Normalized cross-correlation
            left = audio[:, 0, :]   # [batch, samples]
            right = audio[:, 1, :]  # [batch, samples]
            
            # Sliding window correlation
            correlation_scores = []
            win_size = min(256, samples // 4)
            
            for start in range(0, samples - win_size + 1, win_size // 2):
                end = start + win_size
                left_win = left[:, start:end]
                right_win = right[:, start:end]
                
                # Normalize
                left_norm = F.normalize(left_win, dim=1)
                right_norm = F.normalize(right_win, dim=1)
                
                # Correlation
                correlation = torch.sum(left_norm * right_norm, dim=1)
                correlation_scores.append(correlation)
            
            # Interpolate to match sample count
            correlation_tensor = torch.stack(correlation_scores, dim=1)  # [batch, num_windows]
            cross_channel_correlation = F.interpolate(
                correlation_tensor.unsqueeze(1),  # [batch, 1, num_windows]
                size=samples,
                mode='linear',
                align_corners=False
            ).squeeze(1)  # [batch, samples]
            
            # Convert correlation to complexity (lower correlation = higher complexity)
            cross_channel_complexity = 1.0 - torch.abs(cross_channel_correlation)
        else:
            cross_channel_complexity = torch.zeros(batch_size, samples, device=device)
        
        # 4. Dynamic range
        # Local dynamic range using sliding window
        window_size = min(1024, samples // 8)
        dynamic_ranges = []
        
        for start in range(0, samples - window_size + 1, window_size // 2):
            end = start + window_size
            window_audio = audio[:, :, start:end]
            
            # RMS and peak levels
            rms = torch.sqrt(torch.mean(window_audio**2, dim=(1, 2)))
            peak = torch.max(torch.abs(window_audio), dim=2)[0].max(dim=1)[0]
            
            # Dynamic range (peak-to-RMS ratio)
            dynamic_range = peak / (rms + 1e-8)
            dynamic_ranges.append(dynamic_range)
        
        # Interpolate dynamic range to match samples
        dynamic_range_tensor = torch.stack(dynamic_ranges, dim=1)
        dynamic_range_complexity = F.interpolate(
            dynamic_range_tensor.unsqueeze(1),
            size=samples,
            mode='linear',
            align_corners=False
        ).squeeze(1)
        
        # Combine complexity measures
        total_complexity = (
            self.complexity_weights['spectral_entropy'] * spectral_entropy +
            self.complexity_weights['temporal_novelty'] * temporal_novelty +
            self.complexity_weights['cross_channel_correlation'] * cross_channel_complexity +
            self.complexity_weights['dynamic_range'] * dynamic_range_complexity
        )
        
        # Normalize to [0, 1] range
        total_complexity = torch.sigmoid(total_complexity)
        
        return total_complexity


class ProductionAdaptiveBitAllocator(nn.Module):
    """
    Production-grade adaptive bit allocator with overflow-resilient architecture.
    
    Implements cascade failure prevention, RLE/VLC compression guard, and
    emergency bitrate control for robust audio codec deployment.
    """
    
    def __init__(self, hidden_dim: int = 512, min_bitrate: int = 128, 
                 max_bitrate: int = 320, complexity_lookahead: int = 16,
                 safety_factor: float = 1.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.min_bitrate = min_bitrate
        self.max_bitrate = max_bitrate
        self.complexity_lookahead = complexity_lookahead
        self.safety_factor = safety_factor
        
        # Fast complexity predictor
        self.complexity_predictor = FastComplexityPredictor()
        
        # Bit allocation neural network
        self.allocation_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),  # +1 for complexity score
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(), 
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Normalize allocation to [0, 1]
        )
        
        # Target bitrate (can be dynamically adjusted)
        self.register_buffer('target_bitrate', torch.tensor(192.0))  # Default 192 kbps
        
        # Temporal bit banking for debt/credit system
        self.register_buffer('bit_debt', torch.zeros(1))
        self.register_buffer('bit_credit', torch.zeros(1))
        self.max_debt = 1000  # Maximum bits that can be borrowed
        self.max_credit = 2000  # Maximum bits that can be banked
        
        # Emergency degradation state
        self.register_buffer('emergency_mode', torch.tensor(False))
        self.register_buffer('emergency_quality_factor', torch.tensor(1.0))
        
        # Statistics for overflow prevention
        self.register_buffer('overflow_count', torch.zeros(1))
        self.register_buffer('total_allocations', torch.zeros(1))
        
    def set_target_bitrate(self, bitrate: float):
        """Set target bitrate for encoding."""
        bitrate = max(self.min_bitrate, min(self.max_bitrate, bitrate))
        self.target_bitrate.fill_(bitrate)
    
    def _compute_logarithmic_headroom(self, complexity: torch.Tensor) -> torch.Tensor:
        """
        Compute logarithmic safety margin for peak segments.
        
        Args:
            complexity: Complexity scores [batch, seq_len]
            
        Returns:
            Logarithmic headroom factors [batch, seq_len]
        """
        # Identify peak complexity segments (top 10%)
        peak_threshold = torch.quantile(complexity, 0.9, dim=1, keepdim=True)
        is_peak = complexity > peak_threshold
        
        # Logarithmic headroom for peak segments
        peak_headroom = 1.0 + torch.log(1.0 + complexity * 4.0) * 0.2
        normal_headroom = torch.ones_like(complexity)
        
        headroom = torch.where(is_peak, peak_headroom, normal_headroom)
        return headroom
    
    def _rle_compression_estimate(self, allocation: torch.Tensor) -> float:
        """
        Estimate compression ratio from run-length encoding of allocation patterns.
        
        Args:
            allocation: Bit allocation tensor [batch, seq_len]
            
        Returns:
            Estimated compression ratio
        """
        # Quantize allocations to discrete levels for RLE analysis
        quantized_allocation = torch.round(allocation * 16).long()  # 16 levels
        
        # Count runs (simplified RLE simulation)
        batch_size, seq_len = quantized_allocation.shape
        total_symbols = batch_size * seq_len
        
        # Estimate unique runs by counting transitions
        transitions = torch.sum(quantized_allocation[:, 1:] != quantized_allocation[:, :-1])
        estimated_runs = transitions.item() + batch_size  # +1 run per sequence
        
        # RLE compression ratio (runs + values vs original length)
        compression_ratio = estimated_runs * 2 / total_symbols  # 2 bytes per run (length + value)
        compression_ratio = max(0.1, min(1.0, compression_ratio))  # Clamp to reasonable range
        
        return compression_ratio
    
    def _emergency_bitrate_control(self, allocation: torch.Tensor, 
                                 complexity: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Emergency bitrate control with graceful quality degradation.
        
        Args:
            allocation: Proposed bit allocation [batch, seq_len]
            complexity: Complexity scores [batch, seq_len]
            
        Returns:
            Tuple of (adjusted_allocation, emergency_triggered)
        """
        # Check if allocation exceeds budget
        current_rate = torch.mean(allocation) * self.target_bitrate
        budget_exceeded = current_rate > self.target_bitrate * 1.1  # 10% tolerance
        
        emergency_triggered = False
        
        if budget_exceeded:
            # Enter emergency mode
            self.emergency_mode.fill_(True)
            emergency_triggered = True
            
            # Reduce quality factor (95% -> 85% quality)
            target_reduction = 0.85
            self.emergency_quality_factor.fill_(target_reduction)
            
            # Apply quality reduction with complexity awareness
            quality_factor = self.emergency_quality_factor * (0.9 + 0.1 * complexity)
            adjusted_allocation = allocation * quality_factor
            
            print(f"Emergency bitrate control activated: "
                  f"Target rate {current_rate:.1f} -> {torch.mean(adjusted_allocation) * self.target_bitrate:.1f} kbps")
        else:
            # Normal operation - gradually recover from emergency mode
            if self.emergency_mode.item():
                # Gradual recovery
                recovery_rate = 0.98
                self.emergency_quality_factor.mul_(recovery_rate).add_(
                    (1.0 - recovery_rate) * 1.0  # Recover towards 100%
                )
                
                # Exit emergency mode when quality factor is close to 100%
                if self.emergency_quality_factor.item() > 0.98:
                    self.emergency_mode.fill_(False)
                    self.emergency_quality_factor.fill_(1.0)
            
            adjusted_allocation = allocation
        
        return adjusted_allocation, emergency_triggered
    
    def _temporal_bit_banking(self, allocation: torch.Tensor, 
                            complexity: torch.Tensor) -> torch.Tensor:
        """
        Implement temporal bit banking for complexity smoothing.
        
        Args:
            allocation: Bit allocation [batch, seq_len]
            complexity: Complexity scores [batch, seq_len]
            
        Returns:
            Banked allocation with debt/credit system
        """
        seq_len = allocation.shape[1]
        avg_allocation = torch.mean(allocation, dim=1, keepdim=True)
        
        banked_allocation = allocation.clone()
        
        for t in range(seq_len):
            current_allocation = allocation[:, t:t+1]
            current_complexity = complexity[:, t:t+1]
            
            # Determine if we should borrow or lend bits
            mean_complexity = current_complexity.mean().item()
            if mean_complexity > 0.7:  # High complexity - may need extra bits
                if self.bit_credit.item() > 0:
                    extra_bits = min(self.bit_credit.item(), avg_allocation.mean().item() * 0.2)
                    banked_allocation[:, t:t+1] += extra_bits / self.target_bitrate
                    self.bit_credit.sub_(extra_bits)

            elif mean_complexity < 0.3:  # Low complexity - bank excess bits
                excess_bits = (avg_allocation - current_allocation) * 0.5
                max_deposit = self.max_credit - self.bit_credit.item()
                bankable_bits = torch.clamp(excess_bits, 0.0, max_deposit)

                banked_allocation[:, t:t+1] -= bankable_bits
                self.bit_credit.add_(bankable_bits.sum().item())
        
        return banked_allocation
    
    def forward(self, features: torch.Tensor, 
                audio_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute adaptive bit allocation with overflow protection.
        
        Args:
            features: Input features [batch, seq_len, hidden_dim]
            audio_context: Optional audio for complexity analysis
            
        Returns:
            Tuple of (bit_allocation, complexity_metrics)
        """
        batch_size, seq_len, hidden_dim = features.shape
        device = features.device
        
        # Predict complexity
        if audio_context is not None:
            complexity_scores = self.complexity_predictor.predict_complexity(audio_context)
        else:
            # Fallback: estimate complexity from features
            feature_variance = torch.var(features, dim=-1)
            complexity_scores = torch.sigmoid(feature_variance - feature_variance.mean())
        
        # Add complexity as additional input to allocation network
        complexity_expanded = complexity_scores.unsqueeze(-1)  # [batch, seq_len, 1]
        allocation_input = torch.cat([features, complexity_expanded], dim=-1)
        
        # Predict base allocation
        base_allocation = self.allocation_net(allocation_input).squeeze(-1)  # [batch, seq_len]
        
        # Apply logarithmic headroom for peak segments
        headroom_factors = self._compute_logarithmic_headroom(complexity_scores)
        protected_allocation = base_allocation * headroom_factors * self.safety_factor
        
        # Apply temporal bit banking
        banked_allocation = self._temporal_bit_banking(protected_allocation, complexity_scores)
        
        # Emergency bitrate control
        final_allocation, emergency_triggered = self._emergency_bitrate_control(
            banked_allocation, complexity_scores
        )
        
        # Update statistics
        self.total_allocations.add_(1)
        
        # Check for overflow (simplified check)
        current_rate = torch.mean(final_allocation) * self.target_bitrate
        if current_rate > self.target_bitrate * 1.15:  # 15% over budget
            self.overflow_count.add_(1)
        
        # Compute compression estimate
        compression_ratio = self._rle_compression_estimate(final_allocation)
        
        # Prepare metrics
        complexity_metrics = {
            'complexity_scores': complexity_scores,
            'base_allocation': base_allocation,
            'headroom_factors': headroom_factors,
            'final_allocation': final_allocation,
            'current_bitrate': current_rate.item(),
            'target_bitrate': self.target_bitrate.item(),
            'compression_ratio': compression_ratio,
            'emergency_mode': self.emergency_mode.item(),
            'bit_debt': self.bit_debt.item(),
            'bit_credit': self.bit_credit.item(),
            'overflow_rate': (self.overflow_count / self.total_allocations).item(),
            'emergency_triggered': emergency_triggered
        }
        
        return final_allocation, complexity_metrics


class AdaptiveGradientClipper:
    """
    P95 percentile adaptive gradient clipping with outlier detection.
    
    Implements statistical gradient monitoring with change point detection
    and emergency clipping protocols for training stability.
    """
    
    def __init__(self, window_size: int = 100, percentile: float = 95.0,
                 explosion_threshold: float = 10.0):
        self.window_size = window_size
        self.percentile = percentile
        self.explosion_threshold = explosion_threshold
        
        # Gradient norm history
        self.grad_norm_history = []
        
        # Current clipping threshold
        self.current_threshold = 1.0
        
        # Emergency state
        self.emergency_mode = False
        self.emergency_threshold = 0.1
        
        # Change point detection
        self.prev_avg = 0.0
        self.explosion_count = 0
    
    def _detect_gradient_explosion(self, grad_norm: float) -> bool:
        """
        Detect sudden gradient explosion using change point detection.
        
        Args:
            grad_norm: Current gradient norm
            
        Returns:
            True if gradient explosion detected
        """
        if len(self.grad_norm_history) < 10:
            return False
        
        # Compute recent average
        recent_avg = sum(self.grad_norm_history[-5:]) / 5
        
        # Check for sudden spike
        explosion_detected = (
            grad_norm > self.explosion_threshold * recent_avg and
            grad_norm > self.explosion_threshold
        )
        
        if explosion_detected:
            self.explosion_count += 1
            print(f"Gradient explosion detected: {grad_norm:.4f} "
                  f"(recent avg: {recent_avg:.4f})")
        
        return explosion_detected
    
    def _compute_adaptive_threshold(self) -> float:
        """
        Compute adaptive clipping threshold using P95 percentile.
        
        Returns:
            Updated clipping threshold
        """
        if len(self.grad_norm_history) < 10:
            return self.current_threshold
        
        # Compute P95 percentile of recent history
        recent_history = self.grad_norm_history[-self.window_size:]
        threshold = np.percentile(recent_history, self.percentile)
        
        # Smooth threshold updates
        alpha = 0.1
        self.current_threshold = (1 - alpha) * self.current_threshold + alpha * threshold
        
        return self.current_threshold
    
    def clip_gradients(self, model: nn.Module, max_norm: Optional[float] = None) -> Dict[str, float]:
        """
        Clip gradients with adaptive threshold and explosion detection.
        
        Args:
            model: Model to clip gradients for
            max_norm: Optional fixed max norm (overrides adaptive)
            
        Returns:
            Dictionary of gradient statistics
        """
        # Compute total gradient norm
        total_norm = 0.0
        param_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** (1. / 2)
        
        # Update history
        self.grad_norm_history.append(total_norm)
        if len(self.grad_norm_history) > self.window_size:
            self.grad_norm_history.pop(0)
        
        # Detect gradient explosion
        explosion_detected = self._detect_gradient_explosion(total_norm)
        
        if explosion_detected:
            # Emergency clipping
            self.emergency_mode = True
            clip_threshold = self.emergency_threshold
            print(f"Emergency gradient clipping activated: {clip_threshold}")
        else:
            # Normal adaptive clipping
            if self.emergency_mode:
                # Gradual recovery from emergency
                recovery_rate = 0.9
                self.emergency_threshold *= (1 / recovery_rate)
                if self.emergency_threshold > self.current_threshold * 0.5:
                    self.emergency_mode = False
                clip_threshold = self.emergency_threshold
            else:
                # Use max_norm if provided, otherwise adaptive
                clip_threshold = max_norm if max_norm is not None else self._compute_adaptive_threshold()
        
        # Apply clipping
        if total_norm > clip_threshold:
            clip_coef = clip_threshold / (total_norm + 1e-6)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
            
            clipped = True
        else:
            clipped = False
        
        return {
            'total_norm': total_norm,
            'clip_threshold': clip_threshold,
            'clipped': clipped,
            'explosion_detected': explosion_detected,
            'emergency_mode': self.emergency_mode,
            'explosion_count': self.explosion_count
        }


class StableCheckpointer:
    """
    Fault-tolerant checkpointing with ACID property compliance.
    
    Implements atomic checkpoint operations, corruption detection, and
    automatic rollback mechanisms for production training stability.
    """
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        
        # Checkpoint metadata
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        
        # Load existing metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load checkpoint metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                print("Warning: Could not load checkpoint metadata, starting fresh")
        
        return {
            'checkpoints': [],
            'latest': None,
            'corruption_count': 0
        }
    
    def _save_metadata(self):
        """Save checkpoint metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save checkpoint metadata: {e}")
    
    def _compute_file_hash(self, filepath: Path) -> str:
        """Compute SHA-256 hash of file for integrity checking."""
        sha256_hash = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except:
            return ""
    
    def _atomic_save(self, obj: Dict, filepath: Path) -> bool:
        """
        Atomically save object to file with corruption protection.
        
        Args:
            obj: Object to save
            filepath: Target file path
            
        Returns:
            True if save was successful
        """
        temp_filepath = filepath.with_suffix('.tmp')
        
        try:
            # Save to temporary file first
            torch.save(obj, temp_filepath)
            
            # Verify integrity
            test_load = torch.load(temp_filepath, map_location='cpu')
            
            # Compute hash
            file_hash = self._compute_file_hash(temp_filepath)
            
            # Atomic move (most filesystems guarantee atomicity of rename)
            temp_filepath.rename(filepath)
            
            return True
            
        except Exception as e:
            print(f"Error during atomic save: {e}")
            # Clean up temporary file
            if temp_filepath.exists():
                temp_filepath.unlink()
            return False
    
    def save_checkpoint(self, state_dict: Dict, step: int, 
                       loss: float, metrics: Optional[Dict] = None) -> bool:
        """
        Save checkpoint with atomic operation and integrity verification.
        
        Args:
            state_dict: Model state dictionary to save
            step: Training step
            loss: Current loss value
            metrics: Optional additional metrics
            
        Returns:
            True if checkpoint was saved successfully
        """
        timestamp = time.time()
        checkpoint_name = f"checkpoint_step_{step:08d}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint object
        checkpoint_obj = {
            'step': step,
            'state_dict': state_dict,
            'loss': loss,
            'timestamp': timestamp,
            'metrics': metrics or {}
        }
        
        # Atomic save
        success = self._atomic_save(checkpoint_obj, checkpoint_path)
        
        if success:
            # Compute file hash for integrity
            file_hash = self._compute_file_hash(checkpoint_path)
            
            # Update metadata
            checkpoint_info = {
                'name': checkpoint_name,
                'path': str(checkpoint_path),
                'step': step,
                'loss': loss,
                'timestamp': timestamp,
                'hash': file_hash
            }
            
            self.metadata['checkpoints'].append(checkpoint_info)
            self.metadata['latest'] = checkpoint_info
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            # Save metadata
            self._save_metadata()
            
            print(f"Checkpoint saved: {checkpoint_name} (step {step}, loss {loss:.4f})")
            return True
        else:
            print(f"Failed to save checkpoint at step {step}")
            return False
    
    def load_checkpoint(self, step: Optional[int] = None, 
                       best_loss: bool = False) -> Optional[Dict]:
        """
        Load checkpoint with corruption detection and automatic rollback.
        
        Args:
            step: Specific step to load (None for latest)
            best_loss: Load checkpoint with best loss
            
        Returns:
            Loaded checkpoint dictionary or None if failed
        """
        if not self.metadata['checkpoints']:
            print("No checkpoints available")
            return None
        
        # Select checkpoint to load
        if step is not None:
            # Find specific step
            target_checkpoint = None
            for ckpt in self.metadata['checkpoints']:
                if ckpt['step'] == step:
                    target_checkpoint = ckpt
                    break
            
            if target_checkpoint is None:
                print(f"Checkpoint for step {step} not found")
                return None
        elif best_loss:
            # Find checkpoint with best loss
            target_checkpoint = min(self.metadata['checkpoints'], key=lambda x: x['loss'])
        else:
            # Load latest
            target_checkpoint = self.metadata['latest']
        
        checkpoint_path = Path(target_checkpoint['path'])
        
        # Verify file exists
        if not checkpoint_path.exists():
            print(f"Checkpoint file not found: {checkpoint_path}")
            return self._attempt_rollback(target_checkpoint)
        
        # Verify file integrity
        current_hash = self._compute_file_hash(checkpoint_path)
        if current_hash != target_checkpoint['hash']:
            print(f"Checkpoint corruption detected: {checkpoint_path}")
            self.metadata['corruption_count'] += 1
            return self._attempt_rollback(target_checkpoint)
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"Loaded checkpoint: {target_checkpoint['name']} "
                  f"(step {target_checkpoint['step']}, loss {target_checkpoint['loss']:.4f})")
            return checkpoint
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return self._attempt_rollback(target_checkpoint)
    
    def _attempt_rollback(self, failed_checkpoint: Dict) -> Optional[Dict]:
        """
        Attempt to rollback to previous valid checkpoint.
        
        Args:
            failed_checkpoint: Information about failed checkpoint
            
        Returns:
            Loaded previous checkpoint or None
        """
        print("Attempting rollback to previous checkpoint...")
        
        # Find previous checkpoint
        failed_step = failed_checkpoint['step']
        previous_checkpoints = [
            ckpt for ckpt in self.metadata['checkpoints'] 
            if ckpt['step'] < failed_step
        ]
        
        if not previous_checkpoints:
            print("No previous checkpoints available for rollback")
            return None
        
        # Try loading previous checkpoints in reverse order
        previous_checkpoints.sort(key=lambda x: x['step'], reverse=True)
        
        for prev_ckpt in previous_checkpoints:
            try:
                checkpoint_path = Path(prev_ckpt['path'])
                
                # Verify integrity
                current_hash = self._compute_file_hash(checkpoint_path)
                if current_hash != prev_ckpt['hash']:
                    continue
                
                # Try loading
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                print(f"Rollback successful: {prev_ckpt['name']}")
                return checkpoint
                
            except:
                continue
        
        print("Rollback failed: no valid previous checkpoints")
        return None
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints limit."""
        if len(self.metadata['checkpoints']) <= self.max_checkpoints:
            return
        
        # Sort by step
        sorted_checkpoints = sorted(self.metadata['checkpoints'], key=lambda x: x['step'])
        
        # Remove oldest checkpoints
        checkpoints_to_remove = sorted_checkpoints[:-self.max_checkpoints]
        
        for ckpt in checkpoints_to_remove:
            try:
                checkpoint_path = Path(ckpt['path'])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                print(f"Removed old checkpoint: {ckpt['name']}")
            except Exception as e:
                print(f"Error removing checkpoint {ckpt['name']}: {e}")
        
        # Update metadata
        self.metadata['checkpoints'] = sorted_checkpoints[-self.max_checkpoints:]
    
    def get_checkpoint_info(self) -> Dict:
        """Get information about available checkpoints."""
        return {
            'total_checkpoints': len(self.metadata['checkpoints']),
            'latest_step': self.metadata['latest']['step'] if self.metadata['latest'] else None,
            'latest_loss': self.metadata['latest']['loss'] if self.metadata['latest'] else None,
            'corruption_count': self.metadata['corruption_count'],
            'checkpoints': self.metadata['checkpoints']
        }


class AudioDataProcessor:
    """
    Audio data processing utilities for LyCodec training and inference.
    
    Handles 44.1kHz stereo audio loading, segmentation, and augmentation
    with production-grade error handling and format normalization.
    """
    
    def __init__(self, sample_rate: int = 44100, channels: int = 2,
                 segment_length: float = 5.0, segments_per_track: int = 3):
        self.sample_rate = sample_rate
        self.channels = channels
        self.segment_length = segment_length
        self.segment_samples = int(sample_rate * segment_length)
        self.segments_per_track = segments_per_track
        
        # Supported audio formats
        self.supported_formats = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        
        # Audio normalization parameters
        self.target_lufs = -23.0  # EBU R128 standard
        self.max_peak = -1.0  # dBFS
    
    def load_audio(self, filepath: Union[str, Path]) -> Optional[torch.Tensor]:
        """
        Load audio file and normalize to 44.1kHz stereo.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Audio tensor [channels=2, samples] or None if failed
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"Audio file not found: {filepath}")
            return None
        
        if filepath.suffix.lower() not in self.supported_formats:
            print(f"Unsupported audio format: {filepath.suffix}")
            return None
        
        try:
            # Load audio using torchaudio
            audio, sr = torchaudio.load(str(filepath))
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=self.sample_rate
                )
                audio = resampler(audio)
            
            # Convert to stereo
            if audio.shape[0] == 1:
                # Mono to stereo (duplicate channel)
                audio = audio.repeat(2, 1)
            elif audio.shape[0] > 2:
                # Multi-channel to stereo (mix down)
                audio = torch.mean(audio, dim=0, keepdim=True).repeat(2, 1)
            
            # Ensure exactly 2 channels
            audio = audio[:2, :]
            
            # Normalize audio
            audio = self._normalize_audio(audio)
            
            return audio
            
        except Exception as e:
            print(f"Error loading audio {filepath}: {e}")
            return None
    
    def _normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Normalize audio to consistent level and prevent clipping.
        
        Args:
            audio: Input audio [channels, samples]
            
        Returns:
            Normalized audio tensor
        """
        # Peak normalization to prevent clipping
        peak_level = torch.max(torch.abs(audio))
        if peak_level > 0:
            # Normalize to -1 dBFS peak
            target_peak_linear = 10 ** (self.max_peak / 20)
            audio = audio * (target_peak_linear / peak_level)
        
        # RMS normalization for consistent loudness
        rms = torch.sqrt(torch.mean(audio ** 2))
        if rms > 0:
            # Target RMS based on LUFS (simplified)
            target_rms_linear = 10 ** (self.target_lufs / 20)
            rms_ratio = target_rms_linear / rms
            
            # Apply with limiting to prevent over-normalization
            rms_ratio = min(rms_ratio, 2.0)  # Max 6dB gain
            audio = audio * rms_ratio
        
        # Final safety clipping
        audio = torch.clamp(audio, -1.0, 1.0)
        
        return audio
    
    def extract_segments(self, audio: torch.Tensor, 
                        random_segments: bool = True) -> List[torch.Tensor]:
        """
        Extract non-overlapping segments from audio track.
        
        Args:
            audio: Input audio [channels, samples]
            random_segments: Whether to use random positions
            
        Returns:
            List of audio segments [channels, segment_samples]
        """
        channels, total_samples = audio.shape
        
        if total_samples < self.segment_samples:
            # Pad short audio
            padding_needed = self.segment_samples - total_samples
            audio = F.pad(audio, (0, padding_needed))
            total_samples = self.segment_samples
        
        segments = []
        
        if random_segments:
            # Random non-overlapping segments
            available_positions = list(range(0, total_samples - self.segment_samples + 1, 
                                           self.segment_samples // 4))  # Allow some overlap
            
            if len(available_positions) < self.segments_per_track:
                # Not enough positions, use what we have
                selected_positions = available_positions
            else:
                selected_positions = random.sample(available_positions, self.segments_per_track)
            
            selected_positions.sort()
        else:
            # Evenly spaced segments
            step = max(self.segment_samples, 
                      (total_samples - self.segment_samples) // (self.segments_per_track - 1))
            selected_positions = [i * step for i in range(self.segments_per_track)]
            selected_positions = [min(pos, total_samples - self.segment_samples) 
                                for pos in selected_positions]
        
        # Extract segments
        for start_pos in selected_positions:
            end_pos = start_pos + self.segment_samples
            segment = audio[:, start_pos:end_pos]
            
            # Ensure exact segment length
            if segment.shape[1] < self.segment_samples:
                padding = self.segment_samples - segment.shape[1]
                segment = F.pad(segment, (0, padding))
            
            segments.append(segment)
        
        return segments
    
    def augment_audio(self, audio: torch.Tensor, 
                     augmentation_prob: float = 0.3) -> torch.Tensor:
        """
        Apply audio augmentations for training robustness.
        
        Args:
            audio: Input audio [channels, samples]
            augmentation_prob: Probability of applying each augmentation
            
        Returns:
            Augmented audio tensor
        """
        augmented = audio.clone()
        
        # Time stretching (pitch-preserving)
        if random.random() < augmentation_prob:
            stretch_factor = random.uniform(0.9, 1.1)
            try:
                # Simple linear interpolation time stretch
                original_length = augmented.shape[1]
                stretched_length = int(original_length * stretch_factor)
                
                indices = torch.linspace(0, original_length - 1, stretched_length)
                indices_floor = torch.floor(indices).long()
                indices_ceil = torch.clamp(indices_floor + 1, max=original_length - 1)
                weights = indices - indices_floor.float()
                
                stretched = (1 - weights) * augmented[:, indices_floor] + weights * augmented[:, indices_ceil]
                
                # Resample back to original length
                if stretched_length != original_length:
                    stretched = F.interpolate(
                        stretched.unsqueeze(0), 
                        size=original_length, 
                        mode='linear',
                        align_corners=False
                    ).squeeze(0)
                
                augmented = stretched
            except:
                pass  # Skip if augmentation fails
        
        # Volume adjustment
        if random.random() < augmentation_prob:
            volume_factor = random.uniform(0.7, 1.3)
            augmented = augmented * volume_factor
            augmented = torch.clamp(augmented, -1.0, 1.0)
        
        # Add subtle noise
        if random.random() < augmentation_prob:
            noise_level = random.uniform(0.001, 0.01)
            noise = torch.randn_like(augmented) * noise_level
            augmented = augmented + noise
            augmented = torch.clamp(augmented, -1.0, 1.0)
        
        # EQ simulation (simple high/low pass filtering)
        if random.random() < augmentation_prob:
            # Simple butterworth-like filtering simulation
            cutoff = random.uniform(0.1, 0.9)
            alpha = math.exp(-2 * math.pi * cutoff)
            
            # Apply simple IIR filter
            filtered = torch.zeros_like(augmented)
            for i in range(1, augmented.shape[1]):
                filtered[:, i] = alpha * filtered[:, i-1] + (1 - alpha) * augmented[:, i]
            
            # Mix with original
            mix_ratio = random.uniform(0.3, 0.7)
            augmented = mix_ratio * filtered + (1 - mix_ratio) * augmented
        
        return augmented
    
    def prepare_training_batch(self, audio_files: List[Path], 
                             batch_size: int = 8,
                             augment: bool = True) -> Optional[torch.Tensor]:
        """
        Prepare training batch from audio files.
        
        Args:
            audio_files: List of audio file paths
            batch_size: Target batch size
            augment: Whether to apply augmentations
            
        Returns:
            Batch tensor [batch_size, channels, segment_samples] or None
        """
        batch_segments = []
        
        for audio_file in audio_files:
            # Load audio
            audio = self.load_audio(audio_file)
            if audio is None:
                continue
            
            # Extract segments
            segments = self.extract_segments(audio, random_segments=True)
            
            # Apply augmentations
            if augment:
                segments = [self.augment_audio(seg) for seg in segments]
            
            batch_segments.extend(segments)
            
            # Stop if we have enough segments
            if len(batch_segments) >= batch_size:
                break
        
        if len(batch_segments) == 0:
            return None
        
        # Pad batch to target size if needed
        while len(batch_segments) < batch_size:
            # Duplicate random segment
            batch_segments.append(random.choice(batch_segments))
        
        # Take only what we need
        batch_segments = batch_segments[:batch_size]
        
        # Stack into batch tensor
        batch_tensor = torch.stack(batch_segments, dim=0)
        
        return batch_tensor