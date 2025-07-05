"""
LyCodec v2.5 Training Script with Continuous Latent Space
========================================================

Production-grade training pipeline for f10c10 continuous compression with
semantic preservation, smooth latent manifolds, and ultra-low bitrate optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import yaml
import argparse
import os
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm

# WandB integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' for experiment tracking.")

# Import LyCodec v2.5 components
from lycodec import (
    LyCodecV25Model, LyCodecV25Config,
    ProductionAdaptiveBitAllocator, AdaptiveGradientClipper,
    StableCheckpointer, AudioDataProcessor,
    SAMPLE_RATE, CHANNELS, SEGMENT_LENGTH, SEGMENT_SAMPLES, 
    COMPRESSION_RATIO, LATENT_CHANNELS, LATENT_LENGTH
)


class LyCodecV25Dataset(Dataset):
    """
    Dataset for LyCodec v2.5 training with continuous latent space.
    
    Optimized for f10c10 compression with semantic preservation.
    """
    
    def __init__(self, audio_dir: str, segment_length: float = 5.0,
                 segments_per_track: int = 3, augment: bool = True,
                 cache_size: int = 800, semantic_augment: bool = True):
        self.audio_dir = Path(audio_dir)
        self.segment_length = segment_length
        self.segments_per_track = segments_per_track
        self.augment = augment
        self.semantic_augment = semantic_augment
        self.cache_size = cache_size
        
        # Audio processor optimized for v2.5
        self.audio_processor = AudioDataProcessor(
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            segment_length=segment_length,
            segments_per_track=segments_per_track
        )
        
        # Find all audio files
        self.audio_files = self._find_audio_files()
        self.total_segments = len(self.audio_files) * segments_per_track
        
        # Audio cache
        self.audio_cache = {}
        self.cache_order = []
        
        print(f"LyCodec v2.5 Dataset: {len(self.audio_files)} files, "
              f"{self.total_segments} segments, f{COMPRESSION_RATIO} compression")
    
    def _find_audio_files(self) -> List[Path]:
        """Find supported audio files with enhanced filtering."""
        audio_files = []
        supported_exts = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        
        for ext in supported_exts:
            audio_files.extend(self.audio_dir.rglob(f'*{ext}'))
        
        # Enhanced filtering for v2.5
        filtered_files = []
        for file_path in audio_files:
            try:
                file_size = file_path.stat().st_size
                # Larger minimum size for better semantic content
                if file_size > 500000:  # At least 500KB
                    filtered_files.append(file_path)
            except:
                continue
        
        return filtered_files
    
    def _load_audio_cached(self, file_path: Path) -> Optional[torch.Tensor]:
        """Load audio with LRU caching optimized for v2.5."""
        file_key = str(file_path)
        
        if file_key in self.audio_cache:
            self.cache_order.remove(file_key)
            self.cache_order.append(file_key)
            return self.audio_cache[file_key]
        
        audio = self.audio_processor.load_audio(file_path)
        if audio is None:
            return None
        
        # Add to cache
        self.audio_cache[file_key] = audio
        self.cache_order.append(file_key)
        
        # Maintain cache size
        while len(self.audio_cache) > self.cache_size:
            oldest_key = self.cache_order.pop(0)
            del self.audio_cache[oldest_key]
        
        return audio
    
    def _apply_semantic_augmentation(self, segment: torch.Tensor) -> torch.Tensor:
        """Apply semantic-preserving augmentations for v2.5."""
        if not self.semantic_augment:
            return segment
        
        # Time stretching (preserves pitch/semantics)
        if random.random() < 0.2:
            stretch_factor = random.uniform(0.95, 1.05)
            # Simplified time stretch via interpolation
            original_length = segment.shape[-1]
            stretched_length = int(original_length * stretch_factor)
            
            segment_stretched = F.interpolate(
                segment.unsqueeze(0),
                size=stretched_length,
                mode='linear',
                align_corners=False
            ).squeeze(0)
            
            # Crop or pad to original length
            if stretched_length > original_length:
                segment = segment_stretched[:, :original_length]
            else:
                padding = original_length - stretched_length
                segment = F.pad(segment_stretched, (0, padding))
        
        # Gentle EQ (preserves semantic content)
        if random.random() < 0.3:
            # Simple high/low frequency adjustment
            eq_factor = random.uniform(0.9, 1.1)
            # Apply via simple filtering (placeholder)
            segment = segment * eq_factor
        
        return segment
    
    def __len__(self) -> int:
        return self.total_segments
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Determine file and segment
        file_idx = idx // self.segments_per_track
        segment_idx = idx % self.segments_per_track
        
        file_idx = file_idx % len(self.audio_files)
        audio_file = self.audio_files[file_idx]
        
        # Load audio
        audio = self._load_audio_cached(audio_file)
        
        if audio is None:
            # Fallback
            fallback_idx = random.randint(0, len(self.audio_files) - 1)
            audio_file = self.audio_files[fallback_idx]
            audio = self._load_audio_cached(audio_file)
            
            if audio is None:
                audio = torch.zeros(CHANNELS, SEGMENT_SAMPLES)
        
        # Extract segments
        segments = self.audio_processor.extract_segments(
            audio, random_segments=True
        )
        
        segment_idx = segment_idx % len(segments)
        segment = segments[segment_idx]
        
        # Apply augmentations
        if self.augment:
            segment = self.audio_processor.augment_audio(segment)
            segment = self._apply_semantic_augmentation(segment)
        
        return {
            'audio': segment,
            'file_path': str(audio_file),
            'segment_idx': segment_idx,
            'compression_target': COMPRESSION_RATIO  # For monitoring
        }


class LyCodecV25Loss(nn.Module):
    """
    Comprehensive loss function for LyCodec v2.5 continuous training.
    
    Includes semantic preservation, contrastive learning, and information theory losses.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Primary loss weights
        self.reconstruction_weight = config.get('reconstruction_weight', 1.0)
        self.perceptual_weight = config.get('perceptual_weight', 0.1)
        
        # Continuous space loss weights
        self.kl_weight = config.get('kl_weight', 0.1)
        self.semantic_weight = config.get('semantic_weight', 0.3)
        self.contrastive_weight = config.get('contrastive_weight', 0.2)
        self.consistency_weight = config.get('consistency_weight', 0.15)
        
        # Information theory loss weights
        self.information_bottleneck_weight = config.get('information_bottleneck_weight', 0.05)
        self.mutual_information_weight = config.get('mutual_information_weight', 0.02)
        
        # Traditional loss weights (reduced)
        self.bitrate_weight = config.get('bitrate_weight', 0.005)
        
        # Multi-scale STFT for perceptual loss
        self.stft_scales = [
            {'n_fft': 2048, 'hop_length': 512},
            {'n_fft': 1024, 'hop_length': 256},
            {'n_fft': 512, 'hop_length': 128}
        ]
        
        # Cache windows
        for scale in self.stft_scales:
            n_fft = scale['n_fft']
            self.register_buffer(f'hann_window_{n_fft}', torch.hann_window(n_fft))
        
        # Semantic task weights
        self.semantic_task_weights = config.get('semantic_task_weights', {
            'speech_recognition': 0.25,
            'music_classification': 0.25,
            'emotion_recognition': 0.25,
            'speaker_identification': 0.25
        })
    
    def _get_cached_window(self, n_fft: int, device: torch.device) -> torch.Tensor:
        """Get cached Hann window."""
        window_name = f'hann_window_{n_fft}'
        window = getattr(self, window_name)
        if window.device != device:
            window = window.to(device)
            setattr(self, window_name, window)
        return window
    
    def _compute_stft_loss(self, predicted: torch.Tensor, 
                          target: torch.Tensor, stft_params: Dict) -> torch.Tensor:
        """Multi-scale STFT loss."""
        n_fft = stft_params['n_fft']
        hop_length = stft_params['hop_length']
        
        window = self._get_cached_window(n_fft, predicted.device)
        
        # Compute STFT
        pred_stft = torch.stft(
            predicted.view(-1, predicted.shape[-1]),
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True
        )
        
        target_stft = torch.stft(
            target.view(-1, target.shape[-1]),
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True
        )
        
        # Magnitude and phase losses
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        magnitude_loss = F.l1_loss(pred_mag, target_mag)
        
        # Reduced phase loss weight for continuous space
        phase_loss = F.mse_loss(pred_stft.real, target_stft.real) + \
                    F.mse_loss(pred_stft.imag, target_stft.imag)
        phase_loss *= 0.05  # Reduced weight
        
        return magnitude_loss + phase_loss
    
    def _compute_perceptual_loss(self, predicted: torch.Tensor, 
                               target: torch.Tensor) -> torch.Tensor:
        """Multi-scale perceptual loss."""
        total_loss = 0.0
        for scale_params in self.stft_scales:
            scale_loss = self._compute_stft_loss(predicted, target, scale_params)
            total_loss += scale_loss
        return total_loss / len(self.stft_scales)
    
    def _compute_semantic_preservation_loss(self, semantic_outputs: Dict, 
                                          target_audio: torch.Tensor) -> torch.Tensor:
        """Compute semantic preservation loss from multi-task outputs."""
        # This is a simplified implementation
        # In practice, you'd need pre-trained models or ground truth labels
        
        semantic_loss = 0.0
        
        for task_name, task_output in semantic_outputs.items():
            if task_name in self.semantic_task_weights:
                # Placeholder: semantic consistency loss
                # You would implement actual semantic loss based on your tasks
                task_loss = torch.mean(torch.abs(task_output))  # Placeholder
                semantic_loss += self.semantic_task_weights[task_name] * task_loss
        
        return semantic_loss
    
    def _compute_latent_smoothness_loss(self, latent: torch.Tensor) -> torch.Tensor:
        """Compute smoothness loss for continuous latent space."""
        # Temporal smoothness
        temporal_diff = torch.diff(latent, dim=1)  # Along time dimension
        smoothness_loss = torch.mean(temporal_diff ** 2)
        
        # Channel consistency
        channel_var = torch.var(latent, dim=-1)  # Variance across channels
        consistency_loss = torch.mean(channel_var)
        
        return smoothness_loss + 0.1 * consistency_loss
    
    def _compute_interpolation_loss(self, latent: torch.Tensor) -> torch.Tensor:
        """Compute loss to encourage smooth interpolation."""
        batch_size = latent.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=latent.device)
        
        # Random interpolation
        alpha = torch.rand(1, device=latent.device)
        idx1 = torch.randperm(batch_size)[:batch_size//2]
        idx2 = torch.randperm(batch_size)[:batch_size//2]
        
        latent1 = latent[idx1]
        latent2 = latent[idx2]
        
        # Linear interpolation
        interpolated = alpha * latent1 + (1 - alpha) * latent2
        
        # Smoothness of interpolated latent
        interpolation_smoothness = self._compute_latent_smoothness_loss(interpolated)
        
        return interpolation_smoothness
    
    def forward(self, model_output: Dict, target_audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive loss for LyCodec v2.5.
        """
        reconstructed_audio = model_output['reconstructed_audio']
        ultra_latent = model_output['ultra_latent']
        semantic_outputs = model_output['semantic_outputs']
        losses_dict = model_output['losses']
        
        # Primary losses
        reconstruction_loss = F.l1_loss(reconstructed_audio, target_audio)
        perceptual_loss = self._compute_perceptual_loss(reconstructed_audio, target_audio)
        
        # Continuous space losses
        kl_loss = losses_dict.get('kl_loss', torch.tensor(0.0))
        consistency_loss = losses_dict.get('consistency_loss', torch.tensor(0.0))
        bottleneck_loss = losses_dict.get('bottleneck_loss', torch.tensor(0.0))
        contrastive_loss = losses_dict.get('contrastive_loss', torch.tensor(0.0))
        
        # Semantic preservation
        semantic_loss = self._compute_semantic_preservation_loss(semantic_outputs, target_audio)
        
        # Latent space regularization
        smoothness_loss = self._compute_latent_smoothness_loss(ultra_latent)
        interpolation_loss = self._compute_interpolation_loss(ultra_latent)
        
        # Information theory losses
        information_bottleneck_loss = bottleneck_loss  # From model
        mutual_information_loss = torch.tensor(0.0, device=target_audio.device)  # Placeholder
        
        # Traditional bitrate loss (minimal weight)
        bitrate_loss = torch.tensor(0.0, device=target_audio.device)  # Placeholder
        
        # Total loss
        total_loss = (
            self.reconstruction_weight * reconstruction_loss +
            self.perceptual_weight * perceptual_loss +
            self.kl_weight * kl_loss +
            self.semantic_weight * semantic_loss +
            self.contrastive_weight * contrastive_loss +
            self.consistency_weight * (consistency_loss + smoothness_loss) +
            self.information_bottleneck_weight * information_bottleneck_loss +
            self.mutual_information_weight * mutual_information_loss +
            self.bitrate_weight * bitrate_loss +
            0.1 * interpolation_loss  # Interpolation smoothness
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'perceptual_loss': perceptual_loss,
            'kl_loss': kl_loss,
            'semantic_loss': semantic_loss,
            'contrastive_loss': contrastive_loss,
            'consistency_loss': consistency_loss,
            'smoothness_loss': smoothness_loss,
            'interpolation_loss': interpolation_loss,
            'information_bottleneck_loss': information_bottleneck_loss,
            'mutual_information_loss': mutual_information_loss,
            'bitrate_loss': bitrate_loss
        }


class LyCodecV25Trainer:
    """
    Advanced trainer for LyCodec v2.5 with continuous latent space training.
    """
    
    def __init__(self, config: Dict, device_id: int = 0, world_size: int = 1):
        self.config = config
        self.device_id = device_id
        self.world_size = world_size
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        
        # Distributed training setup
        self.distributed = world_size > 1
        if self.distributed:
            self._setup_distributed()
        
        # WandB setup
        self.use_wandb = (WANDB_AVAILABLE and 
                         config.get('wandb', {}).get('enabled', False) and 
                         device_id == 0)
        if self.use_wandb:
            self._setup_wandb()
        
        # Model configuration
        self.model_config = LyCodecV25Config()
        self._update_config_from_yaml()
        
        # Initialize model
        self.model = LyCodecV25Model(self.model_config).to(self.device)
        
        # Log v2.5 specific info
        if self.use_wandb:
            wandb.watch(self.model, log_freq=100)
            wandb.log({
                "model/total_parameters": self.model.get_model_size(),
                "model/memory_usage_mb": self.model.get_memory_usage()['total_mb'],
                "model/compression_ratio": COMPRESSION_RATIO,
                "model/latent_channels": LATENT_CHANNELS,
                "model/latent_length": LATENT_LENGTH
            })
        
        # Distributed model
        if self.distributed:
            self.model = DDP(self.model, device_ids=[device_id])
        
        # Loss function
        self.loss_fn = LyCodecV25Loss(config['loss'])
        
        # Optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Training utilities
        self.gradient_clipper = AdaptiveGradientClipper(
            window_size=config['training'].get('gradient_clip_window', 100),
            percentile=config['training'].get('gradient_clip_percentile', 95.0)
        )
        
        self.checkpointer = StableCheckpointer(
            checkpoint_dir=config['training']['checkpoint_dir'],
            max_checkpoints=config['training'].get('max_checkpoints', 5)
        )
        
        # TensorBoard
        if device_id == 0 and config.get('tensorboard', {}).get('enabled', True):
            self.writer = SummaryWriter(log_dir=config['training'].get('log_dir', 'logs_v25'))
        else:
            self.writer = None
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Memory optimization
        self.gradient_checkpointing = config['training'].get('gradient_checkpointing', True)
        self.mixed_precision = config['training'].get('mixed_precision', True)
        
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # v2.5 specific metrics
        self.semantic_metrics = {
            'semantic_similarity_history': [],
            'interpolation_quality_history': [],
            'compression_efficiency_history': []
        }
    
    def _setup_wandb(self):
        """Initialize WandB with v2.5 specific configuration."""
        wandb_config = self.config.get('wandb', {})
        
        # v2.5 specific config
        wandb_log_config = {
            # Model architecture
            'model_version': '2.5',
            'compression_ratio': COMPRESSION_RATIO,
            'latent_channels': LATENT_CHANNELS,
            'latent_length': LATENT_LENGTH,
            'model_hidden_dim': self.config['model'].get('hidden_dim', 512),
            'semantic_dim': self.config['model'].get('semantic_dim', 256),
            
            # Training settings
            'batch_size': self.config['data']['batch_size'],
            'learning_rate': self.config['training']['optimizer']['learning_rate'],
            'num_epochs': self.config['training']['num_epochs'],
            'sample_rate': SAMPLE_RATE,
            'segment_length': SEGMENT_LENGTH,
            
            # v2.5 specific settings
            'kl_weight': self.config['model'].get('kl_weight', 0.0001),
            'semantic_weight': self.config['loss'].get('semantic_weight', 0.3),
            'contrastive_weight': self.config['loss'].get('contrastive_weight', 0.2),
            'beta_vae': self.config['model'].get('beta_vae', True),
            
            # Hardware
            'world_size': self.world_size,
            'mixed_precision': self.mixed_precision,
        }
        
        wandb.init(
            project=wandb_config.get('project', 'lycodec-v2.5-continuous'),
            name=wandb_config.get('run_name', None),
            config=wandb_log_config,
            tags=wandb_config.get('tags', ['continuous-latent', 'semantic-preservation']),
            notes=wandb_config.get('notes', 'LyCodec v2.5 continuous architecture training'),
            group=wandb_config.get('group', 'continuous-architecture'),
            resume=wandb_config.get('resume', False)
        )
    
    def _setup_distributed(self):
        """Setup distributed training."""
        if 'RANK' not in os.environ:
            os.environ['RANK'] = str(self.device_id)
        if 'WORLD_SIZE' not in os.environ:
            os.environ['WORLD_SIZE'] = str(self.world_size)
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '12356'  # Different port for v2.5
        
        dist.init_process_group(backend='nccl')
    
    def _update_config_from_yaml(self):
        """Update model config from YAML."""
        model_cfg = self.config.get('model', {})
        
        # Update config fields
        for field in ['hidden_dim', 'num_layers', 'num_attention_heads', 
                     'semantic_dim', 'kl_weight', 'beta_vae']:
            if field in model_cfg:
                setattr(self.model_config, field, model_cfg[field])
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer for v2.5."""
        opt_config = self.config['training']['optimizer']
        
        if opt_config['type'] == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                betas=opt_config.get('betas', (0.9, 0.999)),
                weight_decay=opt_config.get('weight_decay', 0.01),
                eps=opt_config.get('eps', 1e-8)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config['type']}")
        
        return optimizer
    
    def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler."""
        sched_config = self.config['training']['scheduler']
        
        if sched_config['type'] == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config['T_max'],
                eta_min=sched_config.get('eta_min', 1e-6)
            )
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1.0
            )
        
        return scheduler
    
    def _compute_semantic_metrics(self, model_output: Dict, target_audio: torch.Tensor) -> Dict[str, float]:
        """Compute v2.5 specific semantic metrics."""
        ultra_latent = model_output['ultra_latent']
        
        metrics = {}
        
        # Latent smoothness
        temporal_diff = torch.diff(ultra_latent, dim=1)
        smoothness = 1.0 / (1.0 + torch.mean(temporal_diff ** 2).item())
        metrics['latent_smoothness'] = smoothness
        
        # Compression efficiency
        target_size = target_audio.numel()
        compressed_size = ultra_latent.numel()
        efficiency = target_size / compressed_size
        metrics['compression_efficiency'] = efficiency
        
        # Semantic consistency (placeholder)
        semantic_outputs = model_output.get('semantic_outputs', {})
        if semantic_outputs:
            consistency = sum(torch.mean(torch.abs(output)).item() 
                            for output in semantic_outputs.values()) / len(semantic_outputs)
            metrics['semantic_consistency'] = consistency
        
        return metrics
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step for v2.5 with continuous space."""
        start_time = time.time()
        
        audio = batch['audio'].to(self.device)
        
        # Enable gradient checkpointing
        if self.gradient_checkpointing:
            self.model.train()
        
        # Forward pass with mixed precision
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                model_output = self.model(audio, training=True)
                loss_dict = self.loss_fn(model_output, audio)
                total_loss = loss_dict['total_loss']
        else:
            model_output = self.model(audio, training=True)
            loss_dict = self.loss_fn(model_output, audio)
            total_loss = loss_dict['total_loss']
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.mixed_precision:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_stats = self.gradient_clipper.clip_gradients(self.model)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            grad_stats = self.gradient_clipper.clip_gradients(self.model)
            self.optimizer.step()
        
        self.scheduler.step()
        
        # Compute metrics
        batch_time = time.time() - start_time
        semantic_metrics = self._compute_semantic_metrics(model_output, audio)
        
        self.step += 1
        
        # Compile step statistics
        step_stats = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': loss_dict['reconstruction_loss'].item(),
            'perceptual_loss': loss_dict['perceptual_loss'].item(),
            'kl_loss': loss_dict['kl_loss'].item(),
            'semantic_loss': loss_dict['semantic_loss'].item(),
            'contrastive_loss': loss_dict['contrastive_loss'].item(),
            'consistency_loss': loss_dict['consistency_loss'].item(),
            'smoothness_loss': loss_dict['smoothness_loss'].item(),
            'interpolation_loss': loss_dict['interpolation_loss'].item(),
            'learning_rate': self.scheduler.get_last_lr()[0],
            'batch_time': batch_time,
            'grad_norm': grad_stats['total_norm'],
            'grad_clipped': grad_stats['clipped']
        }
        
        # Add semantic metrics
        step_stats.update(semantic_metrics)
        
        # Update semantic metric history
        self.semantic_metrics['semantic_similarity_history'].append(
            semantic_metrics.get('semantic_consistency', 0.0)
        )
        self.semantic_metrics['compression_efficiency_history'].append(
            semantic_metrics.get('compression_efficiency', 0.0)
        )
        
        return step_stats
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation with v2.5 metrics."""
        self.model.eval()
        val_losses = []
        val_components = {
            'reconstruction': [], 'perceptual': [], 'kl': [],
            'semantic': [], 'contrastive': [], 'consistency': []
        }
        semantic_metrics_list = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", disable=self.device_id != 0):
                audio = batch['audio'].to(self.device)
                
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        model_output = self.model(audio, training=False)
                        loss_dict = self.loss_fn(model_output, audio)
                else:
                    model_output = self.model(audio, training=False)
                    loss_dict = self.loss_fn(model_output, audio)
                
                val_losses.append(loss_dict['total_loss'].item())
                
                # Component losses
                val_components['reconstruction'].append(loss_dict['reconstruction_loss'].item())
                val_components['perceptual'].append(loss_dict['perceptual_loss'].item())
                val_components['kl'].append(loss_dict['kl_loss'].item())
                val_components['semantic'].append(loss_dict['semantic_loss'].item())
                val_components['contrastive'].append(loss_dict['contrastive_loss'].item())
                val_components['consistency'].append(loss_dict['consistency_loss'].item())
                
                # Semantic metrics
                semantic_metrics = self._compute_semantic_metrics(model_output, audio)
                semantic_metrics_list.append(semantic_metrics)
        
        # Compute averages
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_components = {f'val_{k}_loss': sum(v) / len(v) for k, v in val_components.items()}
        
        # Average semantic metrics
        if semantic_metrics_list:
            avg_semantic_metrics = {}
            for key in semantic_metrics_list[0].keys():
                avg_semantic_metrics[f'val_{key}'] = sum(
                    m[key] for m in semantic_metrics_list
                ) / len(semantic_metrics_list)
            avg_components.update(avg_semantic_metrics)
        
        return {'val_loss': avg_val_loss, **avg_components}
    
    def _log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ""):
        """Log metrics to monitoring systems."""
        if step is None:
            step = self.step
        
        # TensorBoard
        if self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'{prefix}{key}' if prefix else key, value, step)
        
        # WandB
        if self.use_wandb:
            wandb_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_name = f'{prefix}{key}' if prefix else key
                    wandb_metrics[metric_name] = value
            
            if wandb_metrics:
                wandb.log(wandb_metrics, step=step)
    
    def train_epoch(self, train_loader: DataLoader, 
                   val_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        epoch_stats = []
        
        if self.device_id == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {self.epoch} (v2.5)")
        else:
            pbar = train_loader
        
        for batch_idx, batch in enumerate(pbar):
            step_stats = self.train_step(batch)
            epoch_stats.append(step_stats)
            
            # Update progress bar
            if self.device_id == 0:
                pbar.set_postfix({
                    'loss': f"{step_stats['total_loss']:.4f}",
                    'recon': f"{step_stats['reconstruction_loss']:.4f}",
                    'sem': f"{step_stats['semantic_loss']:.4f}",
                    'smooth': f"{step_stats['latent_smoothness']:.3f}",
                    'comp': f"{step_stats['compression_efficiency']:.1f}x"
                })
            
            # Logging
            if self.step % self.config['training']['log_interval'] == 0:
                self._log_metrics(step_stats, prefix='train/')
            
            # Checkpointing
            if (self.device_id == 0 and 
                self.step % self.config['training']['checkpoint_interval'] == 0):
                
                checkpoint_state = {
                    'model_state_dict': (self.model.module if self.distributed 
                                       else self.model).state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'step': self.step,
                    'epoch': self.epoch,
                    'config': self.config,
                    'semantic_metrics': self.semantic_metrics
                }
                
                if self.mixed_precision:
                    checkpoint_state['scaler_state_dict'] = self.scaler.state_dict()
                
                self.checkpointer.save_checkpoint(
                    checkpoint_state, 
                    self.step, 
                    step_stats['total_loss']
                )
        
        # Epoch averages
        epoch_avg = {}
        for key in epoch_stats[0].keys():
            if isinstance(epoch_stats[0][key], (int, float)):
                epoch_avg[f'epoch_{key}'] = sum(s[key] for s in epoch_stats) / len(epoch_stats)
        
        # Validation
        if val_loader and self.device_id == 0:
            val_stats = self.validate(val_loader)
            epoch_avg.update(val_stats)
            
            self._log_metrics(val_stats, step=self.epoch, prefix='epoch/')
            
            # Best model tracking
            if val_stats['val_loss'] < self.best_loss:
                self.best_loss = val_stats['val_loss']
                
                best_checkpoint_state = {
                    'model_state_dict': (self.model.module if self.distributed 
                                       else self.model).state_dict(),
                    'step': self.step,
                    'epoch': self.epoch,
                    'best_loss': self.best_loss,
                    'config': self.config
                }
                
                best_model_path = Path(self.config['training']['checkpoint_dir']) / 'best_model_v25.pt'
                torch.save(best_checkpoint_state, best_model_path)
                
                if self.use_wandb:
                    wandb.log({'best_val_loss': self.best_loss}, step=self.epoch)
                    
                    # Save model artifact
                    artifact = wandb.Artifact(f'lycodec_v25_epoch_{self.epoch}', type='model')
                    artifact.add_file(str(best_model_path))
                    wandb.log_artifact(artifact)
        
        self.epoch += 1
        return epoch_avg
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              num_epochs: int = 120):
        """Main training loop for v2.5."""
        if self.device_id == 0:
            print(f"Starting LyCodec v2.5 training for {num_epochs} epochs")
            print(f"Model parameters: {self.model.get_model_size():,}")
            print(f"Compression ratio: f{COMPRESSION_RATIO} ({COMPRESSION_RATIO}:1)")
            print(f"Latent shape: [{LATENT_CHANNELS}, {LATENT_LENGTH}]")
            if self.use_wandb:
                print(f"WandB run: {wandb.run.url}")
        
        try:
            for epoch in range(num_epochs):
                if self.distributed and hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)
                
                epoch_stats = self.train_epoch(train_loader, val_loader)
                
                if self.device_id == 0:
                    print(f"Epoch {epoch} completed:")
                    for key, value in epoch_stats.items():
                        if 'loss' in key or 'semantic' in key or 'compression' in key:
                            print(f"  {key}: {value:.4f}")
                    
                    self._log_metrics(epoch_stats, step=epoch, prefix='epoch/')
        
        except KeyboardInterrupt:
            if self.device_id == 0:
                print("Training interrupted")
        except Exception as e:
            if self.device_id == 0:
                print(f"Training failed: {e}")
            raise
        finally:
            if self.device_id == 0:
                print("LyCodec v2.5 training completed!")
                if self.writer:
                    self.writer.close()
                if self.use_wandb:
                    wandb.finish()


def create_data_loaders(config: Dict, world_size: int = 1, 
                       rank: int = 0) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create data loaders for v2.5."""
    data_config = config['data']
    
    # Training dataset
    train_dataset = LyCodecV25Dataset(
        audio_dir=data_config['train_dir'],
        segment_length=SEGMENT_LENGTH,
        segments_per_track=data_config.get('segments_per_track', 3),
        augment=data_config.get('augment', True),
        cache_size=data_config.get('cache_size', 800),
        semantic_augment=True  # v2.5 feature
    )
    
    # Distributed sampler
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    
    # Training loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    # Validation loader
    val_loader = None
    if 'val_dir' in data_config and Path(data_config['val_dir']).exists():
        val_dataset = LyCodecV25Dataset(
            audio_dir=data_config['val_dir'],
            segment_length=SEGMENT_LENGTH,
            segments_per_track=data_config.get('segments_per_track', 3),
            augment=False,
            cache_size=data_config.get('cache_size', 400),
            semantic_augment=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=data_config['batch_size'],
            shuffle=False,
            num_workers=data_config.get('num_workers', 2),
            pin_memory=True
        )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='LyCodec v2.5 Continuous Training')
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['wandb']['config_path'] = args.config
    
    # Distributed setup
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = args.local_rank
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_id = local_rank
    else:
        device_id = 0
    
    # Set seeds
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config, world_size, rank)
    
    # Initialize trainer
    trainer = LyCodecV25Trainer(config, device_id, world_size)
    
    # Resume if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs']
    )


if __name__ == '__main__':
    main()