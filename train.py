#!/usr/bin/env python3
"""
LyCodec Training Script v2.0 - V100×4 ACCELERATE VERSION
Optimized for V100×4 16GB setup with fixed Triton issues and proper 4-GPU distributed training
"""

import os
import sys
import yaml
import argparse
import random
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
import soundfile as sf
import numpy as np

# IMPROVED: Use Accelerate for 4-GPU distributed training
try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    # Try to import DistributedDataParallelKwargs - may not be available in all versions
    try:
        from accelerate.utils import DistributedDataParallelKwargs
        HAS_DDP_KWARGS = True
    except ImportError:
        HAS_DDP_KWARGS = False
        print("ℹ️ DistributedDataParallelKwargs not available - using basic Accelerator setup")
    
    HAS_ACCELERATE = True
    print("✅ Accelerate available for distributed training")
except ImportError:
    print("❌ Accelerate not available. Please install with: pip install accelerate")
    print("Run: accelerate config  # to setup 4-GPU distributed training")
    sys.exit(1)

from lycodec.training import LyCodecTrainer
from lycodec.audio import normalize_audio, high_quality_resample, create_deterministic_seed, get_optimal_pin_memory

class AudioDataset(torch.utils.data.Dataset):
    """
    Dataset for audio files with 5-second sampling (3 samples per track) - V100×4 OPTIMIZED
    """
    def __init__(self, 
                 data_dir: str,
                 segment_length: int = 220500,  # 5 seconds at 44.1kHz
                 samples_per_track: int = 3,
                 file_limit: int = None,
                 normalize_method: str = 'rms',
                 sample_rate: int = 44100,
                 rank: int = 0,
                 world_size: int = 1):
        
        self.data_dir = Path(data_dir)
        self.segment_length = segment_length
        self.samples_per_track = samples_per_track
        self.normalize_method = normalize_method
        self.sample_rate = sample_rate
        self.rank = rank
        self.world_size = world_size
        
        # Find all audio files
        audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aiff', '.au']
        self.audio_files = []
        
        for ext in audio_extensions:
            # Case insensitive search
            self.audio_files.extend(list(self.data_dir.glob(f'**/*{ext}')))
            self.audio_files.extend(list(self.data_dir.glob(f'**/*{ext.upper()}')))
        
        # Remove duplicates and sort for consistency across GPUs
        self.audio_files = sorted(list(set(self.audio_files)))
        
        # Only log from main process
        if self.rank == 0:
            print(f"📁 Found {len(self.audio_files)} audio files")
        
        # Use all files if file_limit is None
        if file_limit is not None and len(self.audio_files) > file_limit:
            if self.rank == 0:
                print(f"📊 Limiting to {file_limit} audio files (configurable in config.yaml)")
            self.audio_files = self.audio_files[:file_limit]
        elif self.rank == 0:
            print(f"📊 Using all {len(self.audio_files)} audio files")
        
        # Create sample list (3 samples per track)
        self.samples = []
        for file_path in self.audio_files:
            for i in range(samples_per_track):
                self.samples.append((file_path, i))
        
        if self.rank == 0:
            print(f"🎵 Total samples: {len(self.samples)} (distributed across {self.world_size} GPUs)")
            print(f"📊 Samples per GPU: ~{len(self.samples) // self.world_size}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, sample_idx = self.samples[idx]
        
        try:
            # Load audio
            audio, sr = sf.read(str(file_path), always_2d=True)
            
            # High-quality resampling if needed
            if sr != self.sample_rate:
                if self.rank == 0 and sample_idx == 0:  # Log once per file from main process
                    print(f"🔄 Resampling {file_path.name} from {sr}Hz to {self.sample_rate}Hz")
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
                # Deterministic sampling based on file path and sample index
                deterministic_seed = create_deterministic_seed(str(file_path) + str(sample_idx))
                np.random.seed(deterministic_seed % (2**32))
                
                max_start = total_samples - self.segment_length
                start_idx = np.random.randint(0, max_start + 1)
                audio = audio[:, start_idx:start_idx + self.segment_length]
            
            # Normalize with specified method
            audio_tensor = torch.from_numpy(audio).float()
            audio_tensor = normalize_audio(audio_tensor, method=self.normalize_method)
            
            return {
                'audio': audio_tensor,
                'filename': str(file_path.name),
                'sample_idx': sample_idx,
                'original_sr': sr
            }
            
        except Exception as e:
            if self.rank == 0:
                print(f"⚠️ Error loading {file_path}: {e}")
            # Return silence as fallback
            audio = np.zeros((2, self.segment_length), dtype=np.float32)
            return {
                'audio': torch.from_numpy(audio),
                'filename': 'error',
                'sample_idx': 0,
                'original_sr': self.sample_rate
            }

def setup_4gpu_accelerator(config):
    """Setup Accelerator for V100×4 distributed training with maximum compatibility"""
    
    # FIXED: Simple and compatible Accelerator setup
    accelerator_kwargs = {
        'mixed_precision': 'fp16' if config['training']['mixed_precision'] else 'no',
        'gradient_accumulation_steps': config['training']['accumulate_grad_batches'],
    }
    
    # Add optional DDP kwargs only if available
    if HAS_DDP_KWARGS:
        try:
            ddp_kwargs = DistributedDataParallelKwargs(
                find_unused_parameters=False,
                broadcast_buffers=True,
                bucket_cap_mb=25
            )
            accelerator_kwargs['kwargs_handlers'] = [ddp_kwargs]
        except Exception as e:
            print(f"⚠️ DDP kwargs setup failed: {e}, using basic setup")
    
    # Try to add other optional parameters safely
    optional_params = {
        'project_dir': config['training']['checkpoint_dir'],
    }
    
    # Add wandb integration if available and enabled
    if config.get('wandb', {}).get('enabled', True):
        try:
            optional_params['log_with'] = 'wandb'
        except Exception:
            print("ℹ️ WandB integration not available in this Accelerate version")
    
    # Add optional parameters one by one to avoid version issues
    for key, value in optional_params.items():
        try:
            accelerator_kwargs[key] = value
        except Exception:
            print(f"ℹ️ Parameter '{key}' not supported in this Accelerate version")
    
    # Create Accelerator with safe parameters
    accelerator = Accelerator(**accelerator_kwargs)
    
    # Verify 4 GPU setup
    gpu_info = f"num_processes={accelerator.num_processes}, mixed_precision={accelerator.mixed_precision}"
    
    if accelerator.num_processes != 4:
        print(f"⚠️ Expected 4 GPUs, but got {accelerator.num_processes}")
        print("💡 To setup 4-GPU training, run: accelerate config")
        print("💡 Select 'Multi-GPU' and specify 4 GPUs")
        print(f"💡 Current setup: {gpu_info}")
        if accelerator.num_processes == 1:
            print("💡 Currently running in single-GPU mode")
    else:
        print(f"✅ V100×4 setup verified: {gpu_info}")
    
    if accelerator.is_main_process:
        print(f"🚀 V100×4 Training Setup:")
        print(f"   📊 Number of processes: {accelerator.num_processes}")
        print(f"   🔄 Mixed precision: {accelerator.mixed_precision}")
        print(f"   💾 Device: {accelerator.device}")
        print(f"   🔢 Gradient accumulation: {config['training']['accumulate_grad_batches']}")
        effective_batch = config['training']['batch_size'] * config['training']['accumulate_grad_batches'] * accelerator.num_processes
        print(f"   📦 Effective batch size: {effective_batch}")
    
    return accelerator

def setup_data_loader(config: Dict[str, Any], accelerator: Accelerator):
    """Setup data loader with V100×4 optimization"""
    sample_rate = config.get('audio', {}).get('sample_rate', 44100)
    file_limit = config['data'].get('file_limit', None)
    
    dataset = AudioDataset(
        data_dir=config['data']['data_dir'],
        segment_length=int(sample_rate * config['data']['segment_seconds']),
        samples_per_track=config['data']['samples_per_track'],
        file_limit=file_limit,
        normalize_method=config.get('audio', {}).get('normalize_method', 'rms'),
        sample_rate=sample_rate,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes
    )
    
    # V100 optimized DataLoader settings
    pin_memory = get_optimal_pin_memory()
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,  # Accelerate handles distributed sampling
        num_workers=config['training']['num_workers'],
        pin_memory=pin_memory,
        drop_last=True,  # Important for distributed training
        persistent_workers=True if config['training']['num_workers'] > 0 else False,
        prefetch_factor=2 if config['training']['num_workers'] > 0 else None  # Only set when multiprocessing
    )
    
    return dataloader

def create_validation_loader(config: Dict[str, Any], accelerator: Accelerator):
    """Create validation data loader for V100×4"""
    val_config = config.copy()
    val_config['data']['samples_per_track'] = 1  # Only 1 sample per track for validation
    val_config['training']['batch_size'] = max(1, config['training']['batch_size'] // 2)  # Smaller batch
    
    return setup_data_loader(val_config, accelerator)

def validate_v100_environment():
    """Validate V100×4 environment setup"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    if gpu_count < 4:
        print(f"⚠️ Expected 4 GPUs, found {gpu_count}")
        print("💡 This may still work but won't use full V100×4 capacity")
    
    # Check if we're on V100s (approximate check)
    for i in range(min(4, gpu_count)):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f"🔧 GPU {i}: {props.name}, Memory: {memory_gb:.1f}GB")
        
        if memory_gb < 15:  # V100 has ~16GB
            print(f"⚠️ GPU {i} has less than 16GB memory")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Train LyCodec v2.0 on V100×4')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Configuration file path')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from specific checkpoint')
    parser.add_argument('--validate-only', action='store_true',
                       help='Run validation only')
    args = parser.parse_args()
    
    # Validate environment
    if not validate_v100_environment():
        print("❌ Environment validation failed")
        sys.exit(1)
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    # Type validation and conversion
    def validate_config(config):
        """Validate and convert config values to proper types"""
        training = config.get('training', {})
        training['learning_rate'] = float(training.get('learning_rate', 1e-4))
        training['batch_size'] = int(training.get('batch_size', 4))
        training['accumulate_grad_batches'] = int(training.get('accumulate_grad_batches', 4))
        training['num_epochs'] = int(training.get('num_epochs', 1000))
        training['num_workers'] = int(training.get('num_workers', 4))
        training['save_every'] = int(training.get('save_every', 50))
        training['mixed_precision'] = bool(training.get('mixed_precision', True))
        training['gradient_checkpointing'] = bool(training.get('gradient_checkpointing', True))
        
        data = config.get('data', {})
        data['segment_seconds'] = float(data.get('segment_seconds', 5.0))
        data['samples_per_track'] = int(data.get('samples_per_track', 3))
        if data.get('file_limit') is not None:
            data['file_limit'] = int(data['file_limit'])
        
        audio = config.get('audio', {})
        audio['sample_rate'] = int(audio.get('sample_rate', 44100))
        
        config['seed'] = int(config.get('seed', 42))
        return config
    
    config = validate_config(config)
    
    # FIXED: Setup V100×4 Accelerator with proper configuration
    accelerator = setup_4gpu_accelerator(config)
    
    # Set random seeds for reproducibility across all GPUs
    seed = config.get('seed', 42)
    set_seed(seed)  # Accelerate's set_seed handles distributed seeding
    
    # Additional deterministic settings
    torch.backends.cudnn.deterministic = not config.get('hardware', {}).get('cudnn_benchmark', True)
    torch.backends.cudnn.benchmark = config.get('hardware', {}).get('cudnn_benchmark', True)
    
    # Override resume checkpoint if specified
    if args.resume:
        config['training']['resume_checkpoint'] = args.resume
    
    # Setup data loader
    dataloader = setup_data_loader(config, accelerator)
    
    # FIXED: Prepare DataLoader with Accelerate for proper distributed training
    dataloader = accelerator.prepare(dataloader)
    
    # Calculate total steps for V100×4 setup
    steps_per_epoch = len(dataloader) // config['training']['accumulate_grad_batches']
    total_steps = steps_per_epoch * config['training']['num_epochs']
    
    if accelerator.is_main_process:
        print(f"📊 Training Configuration:")
        print(f"   🔢 Steps per epoch: {steps_per_epoch}")
        print(f"   🔢 Total steps: {total_steps}")
        print(f"   📦 Batch size per GPU: {config['training']['batch_size']}")
        print(f"   📦 Total batch size: {config['training']['batch_size'] * accelerator.num_processes}")
        print(f"   📦 Effective batch size: {config['training']['batch_size'] * config['training']['accumulate_grad_batches'] * accelerator.num_processes}")
    
    # Setup trainer with V100×4 optimizations
    sample_rate = config.get('audio', {}).get('sample_rate', 44100)
    
    trainer = LyCodecTrainer(
        model_config={
            **config['model'],
            # FIXED: Disable Triton kernels due to compilation issues
            'use_triton': config.get('advanced', {}).get('use_triton_kernels', False)
        },
        learning_rate=float(config['training']['learning_rate']),
        batch_size=int(config['training']['batch_size']),
        accumulate_grad_batches=int(config['training']['accumulate_grad_batches']),
        max_sequence_length=int(sample_rate * config['data']['segment_seconds']),
        use_amp=bool(config['training']['mixed_precision']),
        use_checkpointing=bool(config['training']['gradient_checkpointing']),
        total_steps=total_steps,
        accelerator=accelerator
    )
    
    # Initialize wandb for V100×4 tracking
    if accelerator.is_main_process and config.get('wandb', {}).get('enabled', True):
        wandb_config = {
            'project': config.get('wandb', {}).get('project', 'lycodec-v2-v100x4'),
            'name': f"lycodec-v2-{config.get('seed', 42)}-v100x4",
            'config': {
                **config,
                'num_processes': accelerator.num_processes,
                'effective_batch_size': config['training']['batch_size'] * config['training']['accumulate_grad_batches'] * accelerator.num_processes,
                'gpu_type': 'V100-16GB',
                'total_steps': total_steps,
                'version': '2.0',
                'hardware': 'V100x4',
                'triton_disabled': True  # Track that Triton is disabled
            },
            'tags': config.get('wandb', {}).get('tags', ['lycodec', 'f10c10', 'v100x4', '16gb', 'v2.0']),
            'notes': f"LyCodec v2.0 training on V100×4 16GB - Triton disabled for stability, Accelerate distributed training"
        }
        trainer.setup_wandb(wandb_config)
    
    # Setup validation data if requested
    val_dataloader = None
    if config.get('validation', {}).get('val_split', 0) > 0:
        val_dataloader = create_validation_loader(config, accelerator)
        if accelerator.is_main_process:
            print(f"✅ Validation loader created with {len(val_dataloader)} batches")
    
    # Training loop with V100×4 optimizations
    start_epoch = 0
    best_loss = float('inf')
    
    # Resume from checkpoint if exists
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True)
    
    latest_checkpoint = checkpoint_dir / 'latest.pt'
    if latest_checkpoint.exists():
        start_epoch, losses = trainer.load_checkpoint(latest_checkpoint)
        best_loss = losses.get('total_loss', best_loss)
        if accelerator.is_main_process:
            print(f"🔄 Resumed from epoch {start_epoch}")
    
    # V100×4 Training loop
    avg_losses = {'total_loss': float('inf')}  # Initialize with default values
    epoch = start_epoch  # Initialize epoch for exception handling
    try:
        for epoch in range(start_epoch, config['training']['num_epochs']):
            if accelerator.is_main_process:
                print(f"🚀 Starting epoch {epoch}/{config['training']['num_epochs']}")
                print(f"📊 DataLoader has {len(dataloader)} batches")
                print(f"🔄 Testing first batch...")
            
            # Train epoch
            avg_losses = trainer.train_epoch(dataloader, epoch)
            
            # Validation
            val_losses = {}
            if val_dataloader and (epoch + 1) % config.get('validation', {}).get('val_every', 10) == 0:
                val_losses = trainer.validate(val_dataloader)
                
                if accelerator.is_main_process:
                    print("📊 Validation results:")
                    for key, value in val_losses.items():
                        print(f"   {key}: {value:.6f}")
            
            # Save checkpoint (only on main process)
            if accelerator.is_main_process:
                all_losses = {**avg_losses, **val_losses}
                
                # Save latest
                trainer.save_checkpoint(epoch, all_losses, latest_checkpoint)
                
                # Save best based on training loss (or validation loss if available)
                current_loss = val_losses.get('val_total_loss', avg_losses['total_loss'])
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_checkpoint = checkpoint_dir / 'best.pt'
                    trainer.save_checkpoint(epoch, all_losses, best_checkpoint)
                    print(f"🏆 New best model saved with loss: {best_loss:.6f}")
                
                # Save periodic
                if (epoch + 1) % config['training']['save_every'] == 0:
                    periodic_checkpoint = checkpoint_dir / f'epoch_{epoch:04d}.pt'
                    trainer.save_checkpoint(epoch, all_losses, periodic_checkpoint)
            
            # Centralized logging
            trainer.log_epoch(epoch, avg_losses, val_losses, best_loss)
            
            # V100 memory management - clear cache periodically
            if torch.cuda.is_available() and (epoch + 1) % config.get('hardware', {}).get('empty_cache_every', 20) == 0:
                torch.cuda.empty_cache()
                if accelerator.is_main_process:
                    print("🧹 Cleared CUDA cache")
    
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            print("\n⏹️ Training interrupted by user")
            # Save emergency checkpoint
            emergency_checkpoint = checkpoint_dir / f'interrupted_epoch_{epoch}.pt'
            trainer.save_checkpoint(epoch, avg_losses, emergency_checkpoint)
            print(f"💾 Emergency checkpoint saved: {emergency_checkpoint}")
    
    except Exception as e:
        if accelerator.is_main_process:
            print(f"\n❌ Training failed with error: {e}")
            import traceback
            traceback.print_exc()
        raise
    
    finally:
        # Cleanup
        trainer.cleanup_wandb()
        if accelerator.is_main_process:
            print("🏁 Training completed!")
            print("📊 Final Statistics:")
            print(f"   🏆 Best loss: {best_loss:.6f}")
            print(f"   📅 Total epochs: {epoch + 1}")
            print(f"   💾 Checkpoints saved in: {checkpoint_dir}")

if __name__ == '__main__':
    main()