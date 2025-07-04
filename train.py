"""
LyCodec v2.1 Training Script
============================

Production-grade training pipeline for 44.1kHz stereo audio codec with
V100×4 16GB optimization, 5-second segment processing, and adaptive
bit allocation learning.
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

# Import LyCodec components
from lycodec import (
    LyCodecModel, LyCodecConfig,
    ProductionAdaptiveBitAllocator, AdaptiveGradientClipper,
    StableCheckpointer, AudioDataProcessor,
    SAMPLE_RATE, CHANNELS, SEGMENT_LENGTH, SEGMENT_SAMPLES, SEGMENTS_PER_TRACK
)


class LyCodecDataset(Dataset):
    """
    Dataset for LyCodec training with 5-second segment extraction.
    
    Processes 44.1kHz stereo audio files into non-overlapping 5-second segments
    with 3 segments per track for efficient training data utilization.
    """
    
    def __init__(self, audio_dir: str, segment_length: float = 5.0,
                 segments_per_track: int = 3, augment: bool = True,
                 cache_size: int = 1000):
        self.audio_dir = Path(audio_dir)
        self.segment_length = segment_length
        self.segments_per_track = segments_per_track
        self.augment = augment
        self.cache_size = cache_size
        
        # Audio processor
        self.audio_processor = AudioDataProcessor(
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            segment_length=segment_length,
            segments_per_track=segments_per_track
        )
        
        # Find all audio files
        self.audio_files = self._find_audio_files()
        
        # Calculate total number of segments
        self.total_segments = len(self.audio_files) * segments_per_track
        
        # LRU cache for loaded audio
        self.audio_cache = {}
        self.cache_order = []
        
        print(f"Dataset initialized: {len(self.audio_files)} files, "
              f"{self.total_segments} total segments")
    
    def _find_audio_files(self) -> List[Path]:
        """Find all supported audio files in the directory."""
        audio_files = []
        supported_exts = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        
        for ext in supported_exts:
            audio_files.extend(self.audio_dir.rglob(f'*{ext}'))
        
        # Filter out very small files
        filtered_files = []
        for file_path in audio_files:
            try:
                file_size = file_path.stat().st_size
                if file_size > 100000:  # At least 100KB
                    filtered_files.append(file_path)
            except:
                continue
        
        print(f"Found {len(filtered_files)} valid audio files")
        return filtered_files
    
    def _load_audio_cached(self, file_path: Path) -> Optional[torch.Tensor]:
        """Load audio with LRU caching."""
        file_key = str(file_path)
        
        # Check cache
        if file_key in self.audio_cache:
            # Move to end (most recently used)
            self.cache_order.remove(file_key)
            self.cache_order.append(file_key)
            return self.audio_cache[file_key]
        
        # Load audio
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
    
    def __len__(self) -> int:
        return self.total_segments
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Determine which file and segment
        file_idx = idx // self.segments_per_track
        segment_idx = idx % self.segments_per_track
        
        # Handle wraparound for safety
        file_idx = file_idx % len(self.audio_files)
        audio_file = self.audio_files[file_idx]
        
        # Load audio
        audio = self._load_audio_cached(audio_file)
        
        if audio is None:
            # Fallback to a different file
            fallback_idx = random.randint(0, len(self.audio_files) - 1)
            audio_file = self.audio_files[fallback_idx]
            audio = self._load_audio_cached(audio_file)
            
            if audio is None:
                # Create silence as last resort
                audio = torch.zeros(CHANNELS, SEGMENT_SAMPLES)
        
        # Extract segments
        segments = self.audio_processor.extract_segments(
            audio, random_segments=True
        )
        
        # Get the requested segment (with wraparound)
        segment_idx = segment_idx % len(segments)
        segment = segments[segment_idx]
        
        # Apply augmentation during training
        if self.augment:
            segment = self.audio_processor.augment_audio(segment)
        
        return {
            'audio': segment,  # [channels, samples]
            'file_path': str(audio_file),
            'segment_idx': segment_idx
        }


class LyCodecLoss(nn.Module):
    """
    Comprehensive loss function for LyCodec training.
    
    Combines reconstruction loss, quantization loss, perceptual loss,
    and bit allocation efficiency for high-quality codec optimization.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Loss weights from config
        self.reconstruction_weight = config.get('reconstruction_weight', 1.0)
        self.quantization_weight = config.get('quantization_weight', 0.25)
        self.perceptual_weight = config.get('perceptual_weight', 0.1)
        self.bitrate_weight = config.get('bitrate_weight', 0.01)
        
        # Perceptual loss (simplified spectral loss)
        self.stft_config = {
            'n_fft': 2048,
            'hop_length': 512,
            'win_length': 2048,
            'window': 'hann'
        }
        
        # Multi-scale STFT losses
        self.stft_scales = [
            {'n_fft': 2048, 'hop_length': 512},
            {'n_fft': 1024, 'hop_length': 256},
            {'n_fft': 512, 'hop_length': 128}
        ]
    
    def _compute_stft_loss(self, predicted: torch.Tensor, 
                          target: torch.Tensor, stft_params: Dict) -> torch.Tensor:
        """Compute STFT-based spectral loss."""
        # Compute STFT for both signals
        pred_stft = torch.stft(
            predicted.view(-1, predicted.shape[-1]),
            n_fft=stft_params['n_fft'],
            hop_length=stft_params['hop_length'],
            window=torch.hann_window(stft_params['n_fft'], device=predicted.device),
            return_complex=True
        )
        
        target_stft = torch.stft(
            target.view(-1, target.shape[-1]),
            n_fft=stft_params['n_fft'],
            hop_length=stft_params['hop_length'],
            window=torch.hann_window(stft_params['n_fft'], device=target.device),
            return_complex=True
        )
        
        # Magnitude loss
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        magnitude_loss = F.l1_loss(pred_mag, target_mag)
        
        # Phase-aware loss (reduced weight)
        pred_real, pred_imag = pred_stft.real, pred_stft.imag
        target_real, target_imag = target_stft.real, target_stft.imag
        
        real_loss = F.mse_loss(pred_real, target_real)
        imag_loss = F.mse_loss(pred_imag, target_imag)
        phase_loss = (real_loss + imag_loss) * 0.1  # Reduced weight for phase
        
        return magnitude_loss + phase_loss
    
    def _compute_perceptual_loss(self, predicted: torch.Tensor, 
                               target: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale perceptual loss."""
        total_loss = 0.0
        
        for scale_params in self.stft_scales:
            scale_loss = self._compute_stft_loss(predicted, target, scale_params)
            total_loss += scale_loss
        
        return total_loss / len(self.stft_scales)
    
    def _compute_bitrate_efficiency_loss(self, metadata: Dict) -> torch.Tensor:
        """Compute bitrate efficiency regularization."""
        if 'bit_allocation' not in metadata:
            return torch.tensor(0.0, device=next(iter(metadata.values())).device)
        
        bit_allocation = metadata['bit_allocation']
        
        # Encourage smooth bit allocation (reduce variance)
        allocation_variance = torch.var(bit_allocation, dim=1)
        smoothness_loss = torch.mean(allocation_variance)
        
        # Encourage efficient bitrate usage
        if 'current_bitrate' in metadata and 'target_bitrate' in metadata:
            current_bitrate = metadata['current_bitrate']
            target_bitrate = metadata['target_bitrate']
            
            # Penalize excessive bitrate usage
            overage_loss = F.relu(current_bitrate - target_bitrate * 1.1)
            efficiency_loss = smoothness_loss + overage_loss
        else:
            efficiency_loss = smoothness_loss
        
        return efficiency_loss
    
    def forward(self, model_output: Dict, target_audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive loss for LyCodec training.
        
        Args:
            model_output: Dictionary containing model outputs
            target_audio: Ground truth audio [batch, channels, samples]
            
        Returns:
            Dictionary of loss components and total loss
        """
        reconstructed_audio = model_output['reconstructed_audio']
        quantization_loss = model_output['quantization_loss']
        metadata = model_output.get('metadata', {})
        
        # 1. Reconstruction loss (time domain)
        reconstruction_loss = F.mse_loss(reconstructed_audio, target_audio)
        
        # 2. Perceptual loss (frequency domain)
        perceptual_loss = self._compute_perceptual_loss(reconstructed_audio, target_audio)
        
        # 3. Bitrate efficiency loss
        bitrate_loss = self._compute_bitrate_efficiency_loss(metadata)
        
        # 4. Total loss
        total_loss = (
            self.reconstruction_weight * reconstruction_loss +
            self.quantization_weight * quantization_loss +
            self.perceptual_weight * perceptual_loss +
            self.bitrate_weight * bitrate_loss
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'quantization_loss': quantization_loss,
            'perceptual_loss': perceptual_loss,
            'bitrate_loss': bitrate_loss
        }


class LyCodecTrainer:
    """
    Production-grade trainer for LyCodec with distributed training support.
    
    Implements V100×4 16GB optimization, stable checkpointing, adaptive
    gradient clipping, and comprehensive monitoring for robust training.
    """
    
    def __init__(self, config: Dict, device_id: int = 0, world_size: int = 1):
        self.config = config
        self.device_id = device_id
        self.world_size = world_size
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        
        # Initialize distributed training if multi-GPU
        self.distributed = world_size > 1
        if self.distributed:
            self._setup_distributed()
        
        # Model configuration
        self.model_config = LyCodecConfig()
        self._update_config_from_yaml()
        
        # Initialize model
        self.model = LyCodecModel(self.model_config).to(self.device)
        
        # Wrap model for distributed training
        if self.distributed:
            self.model = DDP(self.model, device_ids=[device_id])
        
        # Loss function
        self.loss_fn = LyCodecLoss(config['loss'])
        
        # Optimizer with production-grade settings
        self.optimizer = self._setup_optimizer()
        
        # Learning rate scheduler
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
        
        # Monitoring
        if device_id == 0:  # Only on main process
            self.writer = SummaryWriter(
                log_dir=config['training'].get('log_dir', 'logs')
            )
        else:
            self.writer = None
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Memory optimization settings
        self.gradient_checkpointing = config['training'].get('gradient_checkpointing', True)
        self.mixed_precision = config['training'].get('mixed_precision', True)
        
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Performance monitoring
        self.performance_stats = {
            'batch_times': [],
            'memory_usage': [],
            'throughput': []
        }
    
    def _setup_distributed(self):
        """Initialize distributed training."""
        if 'RANK' not in os.environ:
            os.environ['RANK'] = str(self.device_id)
        if 'WORLD_SIZE' not in os.environ:
            os.environ['WORLD_SIZE'] = str(self.world_size)
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '12355'
        
        dist.init_process_group(backend='nccl')
    
    def _update_config_from_yaml(self):
        """Update model config from YAML settings."""
        model_cfg = self.config.get('model', {})
        
        # Update relevant config fields
        if 'hidden_dim' in model_cfg:
            self.model_config.hidden_dim = model_cfg['hidden_dim']
        if 'num_layers' in model_cfg:
            self.model_config.num_layers = model_cfg['num_layers']
        if 'num_attention_heads' in model_cfg:
            self.model_config.num_attention_heads = model_cfg['num_attention_heads']
        if 'harmonics_count' in model_cfg:
            self.model_config.harmonics_count = model_cfg['harmonics_count']
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with production-grade settings."""
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
        elif sched_config['type'] == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=sched_config.get('gamma', 0.95)
            )
        else:
            # Default: no scheduling
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1.0
            )
        
        return scheduler
    
    def _compute_memory_usage(self) -> Dict[str, float]:
        """Compute current GPU memory usage."""
        if not torch.cuda.is_available():
            return {}
        
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
        cached = torch.cuda.memory_reserved(self.device) / 1024**3      # GB
        
        return {
            'allocated_gb': allocated,
            'cached_gb': cached,
            'utilization': allocated / 16.0  # Assuming 16GB V100
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute single training step with memory optimization."""
        start_time = time.time()
        
        # Move batch to device
        audio = batch['audio'].to(self.device)  # [batch, channels, samples]
        
        # Enable gradient checkpointing if configured
        if self.gradient_checkpointing:
            self.model.train()
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
        
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
            
            # Gradient clipping with scaler
            self.scaler.unscale_(self.optimizer)
            grad_stats = self.gradient_clipper.clip_gradients(self.model)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            grad_stats = self.gradient_clipper.clip_gradients(self.model)
            self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        # Performance tracking
        batch_time = time.time() - start_time
        memory_stats = self._compute_memory_usage()
        
        # Update step counter
        self.step += 1
        
        # Prepare return statistics
        step_stats = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': loss_dict['reconstruction_loss'].item(),
            'quantization_loss': loss_dict['quantization_loss'].item(),
            'perceptual_loss': loss_dict['perceptual_loss'].item(),
            'bitrate_loss': loss_dict['bitrate_loss'].item(),
            'learning_rate': self.scheduler.get_last_lr()[0],
            'batch_time': batch_time,
            'grad_norm': grad_stats['total_norm'],
            'grad_clipped': grad_stats['clipped']
        }
        
        # Add memory stats
        step_stats.update({f'memory_{k}': v for k, v in memory_stats.items()})
        
        # Add model-specific metrics
        if 'metadata' in model_output:
            metadata = model_output['metadata']
            if 'current_bitrate' in metadata:
                step_stats['current_bitrate'] = metadata['current_bitrate']
            if 'overflow_rate' in metadata:
                step_stats['overflow_rate'] = metadata['overflow_rate']
        
        return step_stats
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation loop."""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", disable=self.device_id != 0):
                audio = batch['audio'].to(self.device)
                
                # Forward pass
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        model_output = self.model(audio, training=False)
                        loss_dict = self.loss_fn(model_output, audio)
                else:
                    model_output = self.model(audio, training=False)
                    loss_dict = self.loss_fn(model_output, audio)
                
                val_losses.append(loss_dict['total_loss'].item())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        return {'val_loss': avg_val_loss}
    
    def train_epoch(self, train_loader: DataLoader, 
                   val_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_stats = []
        
        # Progress bar for main process only
        if self.device_id == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        else:
            pbar = train_loader
        
        for batch_idx, batch in enumerate(pbar):
            step_stats = self.train_step(batch)
            epoch_stats.append(step_stats)
            
            # Update progress bar
            if self.device_id == 0:
                pbar.set_postfix({
                    'loss': f"{step_stats['total_loss']:.4f}",
                    'lr': f"{step_stats['learning_rate']:.2e}",
                    'mem': f"{step_stats.get('memory_allocated_gb', 0):.1f}GB"
                })
            
            # Log to tensorboard
            if self.writer and self.step % self.config['training']['log_interval'] == 0:
                for key, value in step_stats.items():
                    self.writer.add_scalar(f'train/{key}', value, self.step)
            
            # Checkpoint saving
            if (self.device_id == 0 and 
                self.step % self.config['training']['checkpoint_interval'] == 0):
                
                checkpoint_state = {
                    'model_state_dict': (self.model.module if self.distributed 
                                       else self.model).state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'step': self.step,
                    'epoch': self.epoch,
                    'config': self.config
                }
                
                if self.mixed_precision:
                    checkpoint_state['scaler_state_dict'] = self.scaler.state_dict()
                
                self.checkpointer.save_checkpoint(
                    checkpoint_state, 
                    self.step, 
                    step_stats['total_loss']
                )
        
        # Compute epoch averages
        epoch_avg = {}
        for key in epoch_stats[0].keys():
            if isinstance(epoch_stats[0][key], (int, float)):
                epoch_avg[f'epoch_{key}'] = sum(s[key] for s in epoch_stats) / len(epoch_stats)
        
        # Validation
        if val_loader and self.device_id == 0:
            val_stats = self.validate(val_loader)
            epoch_avg.update(val_stats)
            
            # Update best loss
            if val_stats['val_loss'] < self.best_loss:
                self.best_loss = val_stats['val_loss']
                
                # Save best model
                best_checkpoint_state = {
                    'model_state_dict': (self.model.module if self.distributed 
                                       else self.model).state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'step': self.step,
                    'epoch': self.epoch,
                    'best_loss': self.best_loss,
                    'config': self.config
                }
                
                torch.save(best_checkpoint_state, 
                          Path(self.config['training']['checkpoint_dir']) / 'best_model.pt')
        
        self.epoch += 1
        return epoch_avg
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              num_epochs: int = 100):
        """Main training loop."""
        if self.device_id == 0:
            print(f"Starting training for {num_epochs} epochs")
            print(f"Model parameters: {self.model.get_model_size():,}")
            print(f"Memory usage: {self.model.get_memory_usage()}")
        
        for epoch in range(num_epochs):
            # Set epoch for distributed sampler
            if self.distributed and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            epoch_stats = self.train_epoch(train_loader, val_loader)
            
            # Log epoch results
            if self.device_id == 0:
                print(f"Epoch {epoch} completed:")
                for key, value in epoch_stats.items():
                    print(f"  {key}: {value:.4f}")
                
                # Log to tensorboard
                if self.writer:
                    for key, value in epoch_stats.items():
                        self.writer.add_scalar(f'epoch/{key}', value, epoch)
        
        if self.device_id == 0:
            print("Training completed!")
            self.writer.close()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if self.distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        
        if 'scaler_state_dict' in checkpoint and self.mixed_precision:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Resumed training from step {self.step}, epoch {self.epoch}")


def create_data_loaders(config: Dict, world_size: int = 1, 
                       rank: int = 0) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create training and validation data loaders."""
    data_config = config['data']
    
    # Training dataset
    train_dataset = LyCodecDataset(
        audio_dir=data_config['train_dir'],
        segment_length=SEGMENT_LENGTH,
        segments_per_track=SEGMENTS_PER_TRACK,
        augment=data_config.get('augment', True),
        cache_size=data_config.get('cache_size', 1000)
    )
    
    # Distributed sampler for multi-GPU training
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    
    # Training data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    # Validation dataset (optional)
    val_loader = None
    if 'val_dir' in data_config and Path(data_config['val_dir']).exists():
        val_dataset = LyCodecDataset(
            audio_dir=data_config['val_dir'],
            segment_length=SEGMENT_LENGTH,
            segments_per_track=SEGMENTS_PER_TRACK,
            augment=False,  # No augmentation for validation
            cache_size=data_config.get('cache_size', 500)
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
    parser = argparse.ArgumentParser(description='LyCodec v2.1 Training')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='Local rank for distributed training')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup distributed training
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = args.local_rank
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_id = local_rank
    else:
        device_id = 0
    
    # Set random seeds for reproducibility
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config, world_size, rank)
    
    # Initialize trainer
    trainer = LyCodecTrainer(config, device_id, world_size)
    
    # Resume from checkpoint if provided
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