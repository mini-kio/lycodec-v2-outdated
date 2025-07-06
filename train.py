#!/usr/bin/env python3
"""
LyCodec Training Script - IMPROVED VERSION
Optimized for V100×4 16GB setup with mixed precision and distributed training
"""

import os
import sys
import yaml
import argparse
import random
from pathlib import Path
from typing import Dict, Any

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
import soundfile as sf
import numpy as np

# IMPROVED: Add wandb for experiment tracking
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    print("Warning: wandb not available, logging to console only")
    HAS_WANDB = False

from lycodec.training import LyCodecTrainer
from lycodec.audio import normalize_audio, high_quality_resample, create_deterministic_seed, get_optimal_pin_memory

class AudioDataset(torch.utils.data.Dataset):
    """
    Dataset for audio files with 5-second sampling (3 samples per track) - IMPROVED VERSION
    """
    def __init__(self, 
                 data_dir: str,
                 segment_length: int = 220500,  # 5 seconds at 44.1kHz
                 samples_per_track: int = 3,
                 file_limit: int = 10,
                 normalize_method: str = 'rms',
                 sample_rate: int = 44100):  # IMPROVED: Allow config override
        
        self.data_dir = Path(data_dir)
        self.segment_length = segment_length
        self.samples_per_track = samples_per_track
        self.normalize_method = normalize_method
        self.sample_rate = sample_rate  # IMPROVED: Use config value
        
        # Find all audio files
        audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aiff', '.au']
        self.audio_files = []
        
        for ext in audio_extensions:
            # Case insensitive search
            self.audio_files.extend(list(self.data_dir.glob(f'**/*{ext}')))
            self.audio_files.extend(list(self.data_dir.glob(f'**/*{ext.upper()}')))
        
        # Remove duplicates and sort for consistency
        self.audio_files = sorted(list(set(self.audio_files)))
        
        # Limit number of files
        if len(self.audio_files) > file_limit:
            print(f"Found {len(self.audio_files)} audio files, limiting to {file_limit}")
            self.audio_files = self.audio_files[:file_limit]
        
        print(f"Dataset: {len(self.audio_files)} audio files")
        
        # Create sample list (3 samples per track)
        self.samples = []
        for file_path in self.audio_files:
            for i in range(samples_per_track):
                self.samples.append((file_path, i))
        
        print(f"Total samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, sample_idx = self.samples[idx]
        
        try:
            # Load audio
            audio, sr = sf.read(str(file_path), always_2d=True)
            
            # IMPROVED: Use high-quality resampling instead of np.interp
            if sr != self.sample_rate:
                print(f"Resampling {file_path.name} from {sr}Hz to {self.sample_rate}Hz")
                audio = high_quality_resample(audio.T, sr, self.sample_rate).T
            
            # Convert to stereo if mono
            if audio.shape[1] == 1:
                audio = np.repeat(audio, 2, axis=1)
            elif audio.shape[1] > 2:
                audio = audio[:, :2]  # Take first two channels
            
            # Transpose to [channels, samples]
            audio = audio.T
            
            # Sample 5-second segment
            total_samples = audio.shape[1]
            if total_samples < self.segment_length:
                # Pad with silence if too short
                padding = self.segment_length - total_samples
                audio = np.pad(audio, ((0, 0), (0, padding)), mode='constant')
            else:
                # IMPROVED: Deterministic sampling based on file path and sample index
                deterministic_seed = create_deterministic_seed(str(file_path) + str(sample_idx))
                np.random.seed(deterministic_seed % (2**32))  # Ensure 32-bit seed
                
                max_start = total_samples - self.segment_length
                start_idx = np.random.randint(0, max_start + 1)
                audio = audio[:, start_idx:start_idx + self.segment_length]
            
            # IMPROVED: Normalize with specified method
            audio_tensor = torch.from_numpy(audio).float()
            audio_tensor = normalize_audio(audio_tensor, method=self.normalize_method)
            
            return {
                'audio': audio_tensor,
                'filename': str(file_path.name),
                'sample_idx': sample_idx,
                'original_sr': sr
            }
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return silence as fallback
            audio = np.zeros((2, self.segment_length), dtype=np.float32)
            return {
                'audio': torch.from_numpy(audio),
                'filename': 'error',
                'sample_idx': 0,
                'original_sr': self.sample_rate
            }

def setup_data_loader(config: Dict[str, Any], rank: int = 0, world_size: int = 1):
    """Setup data loader with distributed sampling - IMPROVED"""
    # IMPROVED: Use config values instead of hardcoded constants
    sample_rate = config.get('audio', {}).get('sample_rate', 44100)
    
    dataset = AudioDataset(
        data_dir=config['data']['data_dir'],
        segment_length=int(sample_rate * config['data']['segment_seconds']),
        samples_per_track=config['data']['samples_per_track'],
        file_limit=config['data']['file_limit'],
        normalize_method=config.get('audio', {}).get('normalize_method', 'rms'),
        sample_rate=sample_rate  # IMPROVED: Pass sample_rate from config
    )
    
    # Distributed sampler
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True,
        drop_last=True  # Ensure consistent batch sizes across ranks
    ) if world_size > 1 else None
    
    # IMPROVED: Conditional pin_memory
    pin_memory = get_optimal_pin_memory()
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config['training']['num_workers'],
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=True if config['training']['num_workers'] > 0 else False
    )
    
    return dataloader, sampler

def create_validation_loader(config: Dict[str, Any], rank: int = 0, world_size: int = 1):
    """Create validation data loader"""
    val_config = config.copy()
    val_config['data']['samples_per_track'] = 1  # Only 1 sample per track for validation
    val_config['training']['batch_size'] = max(1, config['training']['batch_size'] // 2)  # Smaller batch for validation
    
    return setup_data_loader(val_config, rank, world_size)

def train_worker(rank: int, world_size: int, config: Dict[str, Any]):
    """Training worker for distributed training - IMPROVED"""
    
    # IMPROVED: Set environment variables for distributed training with random port
    import random
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    
    # IMPROVED: Random port to avoid conflicts in multi-experiment setups
    if 'MASTER_PORT' not in os.environ:
        master_port = 12000 + random.randint(0, 2000)
        os.environ['MASTER_PORT'] = str(master_port)
    
    # IMPROVED: Add NCCL debug for multi-node expansion
    os.environ.setdefault('NCCL_DEBUG', 'INFO')
    
    # IMPROVED: Calculate total_steps dynamically
    dataloader, sampler = setup_data_loader(config, rank, world_size)
    steps_per_epoch = len(dataloader) // config['training']['accumulate_grad_batches']
    total_steps = steps_per_epoch * config['training']['num_epochs']
    
    # IMPROVED: Use config values instead of hardcoded constants
    sample_rate = config.get('audio', {}).get('sample_rate', 44100)
    
    # Setup trainer with calculated total_steps
    trainer = LyCodecTrainer(
        model_config=config['model'],
        learning_rate=config['training']['learning_rate'],
        batch_size=config['training']['batch_size'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        max_sequence_length=int(sample_rate * config['data']['segment_seconds']),
        use_amp=config['training']['mixed_precision'],
        use_checkpointing=config['training']['gradient_checkpointing'],
        world_size=world_size,
        total_steps=total_steps  # IMPROVED: Pass calculated total_steps
    )
    
    # IMPROVED: Initialize wandb for experiment tracking (V100 multi-GPU optimized)
    if trainer.is_main_process and HAS_WANDB:
        # Initialize wandb with V100 hardware info
        wandb.init(
            project="lycodec-v100-training",
            name=f"lycodec-{config.get('seed', 42)}-{world_size}gpu",
            config={
                **config,
                'world_size': world_size,
                'effective_batch_size': config['training']['batch_size'] * config['training']['accumulate_grad_batches'] * world_size,
                'gpu_type': 'V100-16GB',
                'total_steps': total_steps
            },
            tags=['lycodec', 'f10c10', 'v100', f'{world_size}gpu'],
            notes=f"LyCodec training on {world_size}x V100 16GB"
        )
        print(f"WandB initialized: {wandb.run.name}")
    
    # Setup validation data if requested
    val_dataloader = None
    val_sampler = None
    if config.get('validation', {}).get('val_split', 0) > 0:
        val_dataloader, val_sampler = create_validation_loader(config, rank, world_size)
        if trainer.is_main_process:
            print(f"Validation loader created with {len(val_dataloader)} batches")
    
    # Training loop
    start_epoch = 0
    best_loss = float('inf')
    
    # Resume from checkpoint if exists
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True)
    
    latest_checkpoint = checkpoint_dir / 'latest.pt'
    if latest_checkpoint.exists():
        start_epoch, losses = trainer.load_checkpoint(latest_checkpoint)
        best_loss = losses.get('total_loss', best_loss)
        if trainer.is_main_process:
            print(f"Resumed from epoch {start_epoch}")
    
    # Training epochs
    for epoch in range(start_epoch, config['training']['num_epochs']):
        if sampler:
            sampler.set_epoch(epoch)
        
        # Train epoch
        avg_losses = trainer.train_epoch(dataloader, epoch)
        
        # Validation
        val_losses = {}
        if val_dataloader and (epoch + 1) % config.get('validation', {}).get('val_every', 10) == 0:
            if val_sampler:
                val_sampler.set_epoch(epoch)
            val_losses = trainer.validate(val_dataloader)
            
            if trainer.is_main_process:
                print("Validation results:")
                for key, value in val_losses.items():
                    print(f"  {key}: {value:.6f}")
        
        # Save checkpoint
        if trainer.is_main_process:
            # Combine training and validation losses
            all_losses = {**avg_losses, **val_losses}
            
            # Save latest
            trainer.save_checkpoint(epoch, all_losses, latest_checkpoint)
            
            # Save best based on training loss (or validation loss if available)
            current_loss = val_losses.get('val_total_loss', avg_losses['total_loss'])
            if current_loss < best_loss:
                best_loss = current_loss
                best_checkpoint = checkpoint_dir / 'best.pt'
                trainer.save_checkpoint(epoch, all_losses, best_checkpoint)
                print(f"New best model saved with loss: {best_loss:.6f}")
            
            # Save periodic
            if (epoch + 1) % config['training']['save_every'] == 0:
                periodic_checkpoint = checkpoint_dir / f'epoch_{epoch:04d}.pt'
                trainer.save_checkpoint(epoch, all_losses, periodic_checkpoint)
            
            # Log progress
            print(f"Epoch {epoch} completed:")
            for key, value in all_losses.items():
                print(f"  {key}: {value:.6f}")
            print("-" * 50)
            
            # IMPROVED: WandB logging for V100 multi-GPU training
            if HAS_WANDB and wandb.run is not None:
                # Log metrics
                log_dict = {
                    'epoch': epoch,
                    'learning_rate': trainer.scheduler.get_last_lr()[0] if trainer.scheduler else trainer.learning_rate,
                    **{f'train/{k}': v for k, v in avg_losses.items()},
                    **{f'val/{k}': v for k, v in val_losses.items()},
                    'best_loss': best_loss
                }
                
                # Add GPU memory usage for V100 monitoring
                if torch.cuda.is_available():
                    for gpu_id in range(torch.cuda.device_count()):
                        memory_used = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
                        memory_cached = torch.cuda.memory_reserved(gpu_id) / 1024**3  # GB
                        log_dict[f'gpu_{gpu_id}/memory_used_gb'] = memory_used
                        log_dict[f'gpu_{gpu_id}/memory_cached_gb'] = memory_cached
                
                wandb.log(log_dict)
            
            # IMPROVED: Clear cache periodically to prevent memory buildup
            if torch.cuda.is_available() and (epoch + 1) % 10 == 0:
                torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description='Train LyCodec - Improved Version')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Configuration file path')
    parser.add_argument('--gpus', type=int, default=None,
                       help='Number of GPUs to use (default: auto-detect)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from specific checkpoint')
    parser.add_argument('--validate-only', action='store_true',
                       help='Run validation only')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # IMPROVED: Set random seeds more thoroughly
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable for reproducibility, enable for speed
    
    # Override resume checkpoint if specified
    if args.resume:
        config['training']['resume_checkpoint'] = args.resume
    
    # Determine number of GPUs
    if args.gpus is None:
        world_size = torch.cuda.device_count()
    else:
        world_size = min(args.gpus, torch.cuda.device_count())
    
    if world_size == 0:
        print("No CUDA devices available, training on CPU")
        world_size = 1
        train_worker(0, 1, config)
    elif world_size == 1:
        print("Training on single GPU")
        train_worker(0, 1, config)
    else:
        print(f"Training on {world_size} GPUs")
        # IMPROVED: Use spawn for better compatibility
        mp.set_start_method('spawn', force=True)
        mp.spawn(train_worker, args=(world_size, config), nprocs=world_size, join=True)
    
    # IMPROVED: Clean up wandb after training
    if HAS_WANDB and wandb.run is not None:
        wandb.finish()
        print("WandB run finished")

if __name__ == '__main__':
    main()