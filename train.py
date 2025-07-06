#!/usr/bin/env python3
"""
LyCodec Training Script v2.0 - ACCELERATE VERSION
Optimized for V100×4 16GB setup with Accelerate for easy distributed training
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

# IMPROVED: Use Accelerate instead of manual DDP
try:
    from accelerate import Accelerator
    HAS_ACCELERATE = True
except ImportError:
    print("Warning: Accelerate not available. Please install with: pip install accelerate")
    print("Falling back to single GPU training.")
    HAS_ACCELERATE = False
    
    # Dummy Accelerator class for fallback
    class DummyAccelerator:
        def __init__(self, **kwargs):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.is_main_process = True
            self.num_processes = 1
            self.process_index = 0
            self.mixed_precision = 'no'
        
        def prepare(self, *args):
            return args if len(args) > 1 else args[0]
        
        def backward(self, loss):
            loss.backward()
        
        def accumulate(self, model):
            from contextlib import nullcontext
            return nullcontext()  # Proper context manager
        
        def clip_grad_norm_(self, parameters, max_norm):
            torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        
        def get_state_dict(self, model):
            return model.state_dict()
        
        def load_state_dict(self, model, state_dict):
            model.load_state_dict(state_dict)
        
        @property
        def sync_gradients(self):
            return True
    
    Accelerator = DummyAccelerator

from lycodec.training import LyCodecTrainer
from lycodec.audio import normalize_audio, high_quality_resample, create_deterministic_seed, get_optimal_pin_memory

class AudioDataset(torch.utils.data.Dataset):
    """
    Dataset for audio files with 5-second sampling (3 samples per track) - IMPROVED VERSION v2.0
    """
    def __init__(self, 
                 data_dir: str,
                 segment_length: int = 220500,  # 5 seconds at 44.1kHz
                 samples_per_track: int = 3,
                 file_limit: int = None,  # IMPROVED: None means use all files
                 normalize_method: str = 'rms',
                 sample_rate: int = 44100,
                 rank: int = 0):  # IMPROVED: Add rank for distributed logging
        
        self.data_dir = Path(data_dir)
        self.segment_length = segment_length
        self.samples_per_track = samples_per_track
        self.normalize_method = normalize_method
        self.sample_rate = sample_rate
        self.rank = rank  # IMPROVED: Store rank for logging control
        
        # Find all audio files
        audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aiff', '.au']
        self.audio_files = []
        
        for ext in audio_extensions:
            # Case insensitive search
            self.audio_files.extend(list(self.data_dir.glob(f'**/*{ext}')))
            self.audio_files.extend(list(self.data_dir.glob(f'**/*{ext.upper()}')))
        
        # Remove duplicates and sort for consistency
        self.audio_files = sorted(list(set(self.audio_files)))
        
        # IMPROVED: Only log from main process to avoid duplicate messages
        if self.rank == 0:
            print(f"Found {len(self.audio_files)} audio files")
        
        # IMPROVED: Use all files if file_limit is None
        if file_limit is not None and len(self.audio_files) > file_limit:
            if self.rank == 0:  # IMPROVED: Only log from main process
                print(f"Limiting to {file_limit} audio files (configurable in config.yaml)")
            self.audio_files = self.audio_files[:file_limit]
        elif self.rank == 0:
            print(f"Using all {len(self.audio_files)} audio files")
        
        if self.rank == 0:  # IMPROVED: Only log from main process
            print(f"Dataset: {len(self.audio_files)} audio files")
        
        # Create sample list (3 samples per track)
        self.samples = []
        for file_path in self.audio_files:
            for i in range(samples_per_track):
                self.samples.append((file_path, i))
        
        if self.rank == 0:  # IMPROVED: Only log from main process
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
                # IMPROVED: Only log resampling from main process to reduce noise
                if self.rank == 0 and sample_idx == 0:  # Log once per file
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
            # IMPROVED: Only log errors from main process
            if self.rank == 0:
                print(f"Error loading {file_path}: {e}")
            # Return silence as fallback
            audio = np.zeros((2, self.segment_length), dtype=np.float32)
            return {
                'audio': torch.from_numpy(audio),
                'filename': 'error',
                'sample_idx': 0,
                'original_sr': self.sample_rate
            }

def setup_data_loader(config: Dict[str, Any], accelerator: Accelerator):
    """Setup data loader with Accelerate - IMPROVED"""
    # IMPROVED: Use config values instead of hardcoded constants
    sample_rate = config.get('audio', {}).get('sample_rate', 44100)
    
    # IMPROVED: Handle None file_limit to use all files
    file_limit = config['data'].get('file_limit', None)
    
    dataset = AudioDataset(
        data_dir=config['data']['data_dir'],
        segment_length=int(sample_rate * config['data']['segment_seconds']),
        samples_per_track=config['data']['samples_per_track'],
        file_limit=file_limit,  # IMPROVED: Can be None to use all files
        normalize_method=config.get('audio', {}).get('normalize_method', 'rms'),
        sample_rate=sample_rate,  # IMPROVED: Pass sample_rate from config
        rank=accelerator.process_index  # IMPROVED: Use accelerator rank
    )
    
    # IMPROVED: Conditional pin_memory
    pin_memory = get_optimal_pin_memory()
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,  # Accelerate handles distributed sampling automatically
        num_workers=config['training']['num_workers'],
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=True if config['training']['num_workers'] > 0 else False
    )
    
    return dataloader

def create_validation_loader(config: Dict[str, Any], accelerator: Accelerator):
    """Create validation data loader"""
    val_config = config.copy()
    val_config['data']['samples_per_track'] = 1  # Only 1 sample per track for validation
    val_config['training']['batch_size'] = max(1, config['training']['batch_size'] // 2)  # Smaller batch for validation
    
    return setup_data_loader(val_config, accelerator)

def main():
    parser = argparse.ArgumentParser(description='Train LyCodec v2.0 - Accelerate Version')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Configuration file path')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from specific checkpoint')
    parser.add_argument('--validate-only', action='store_true',
                       help='Run validation only')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # IMPROVED: Validate and convert config types to prevent type errors
    def validate_config(config):
        """Validate and convert config values to proper types"""
        # Training config type validation
        training = config.get('training', {})
        training['learning_rate'] = float(training.get('learning_rate', 1e-4))
        training['batch_size'] = int(training.get('batch_size', 4))
        training['accumulate_grad_batches'] = int(training.get('accumulate_grad_batches', 4))
        training['num_epochs'] = int(training.get('num_epochs', 1000))
        training['num_workers'] = int(training.get('num_workers', 4))
        training['save_every'] = int(training.get('save_every', 50))
        training['mixed_precision'] = bool(training.get('mixed_precision', True))
        training['gradient_checkpointing'] = bool(training.get('gradient_checkpointing', True))
        
        # Data config type validation
        data = config.get('data', {})
        data['segment_seconds'] = float(data.get('segment_seconds', 5.0))
        data['samples_per_track'] = int(data.get('samples_per_track', 3))
        if data.get('file_limit') is not None:
            data['file_limit'] = int(data['file_limit'])
        
        # Audio config type validation
        audio = config.get('audio', {})
        audio['sample_rate'] = int(audio.get('sample_rate', 44100))
        
        # Seed validation
        config['seed'] = int(config.get('seed', 42))
        
        return config
    
    config = validate_config(config)
    
    # IMPROVED: Initialize Accelerator (handles all distributed setup automatically)
    if HAS_ACCELERATE:
        accelerator = Accelerator(
            mixed_precision='fp16' if config['training']['mixed_precision'] else 'no',
            gradient_accumulation_steps=config['training']['accumulate_grad_batches'],
            log_with='wandb' if config.get('wandb', {}).get('enabled', True) else None,
            project_dir=config['training']['checkpoint_dir']
        )
    else:
        # Fallback to dummy accelerator for single GPU
        accelerator = Accelerator()
    
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
    
    # IMPROVED: Print info only from main process
    if accelerator.is_main_process:
        print(f"Training on {accelerator.num_processes} GPUs with Accelerate")
        print(f"Mixed precision: {accelerator.mixed_precision}")
        print(f"Device: {accelerator.device}")
    
    # Setup data loader
    dataloader = setup_data_loader(config, accelerator)
    
    # Calculate total steps
    steps_per_epoch = len(dataloader) // config['training']['accumulate_grad_batches']
    total_steps = steps_per_epoch * config['training']['num_epochs']
    
    # IMPROVED: Use config values instead of hardcoded constants
    sample_rate = config.get('audio', {}).get('sample_rate', 44100)
    
    # Setup trainer with Accelerator
    trainer = LyCodecTrainer(
        model_config={
            **config['model'],
            'use_triton': config.get('advanced', {}).get('use_triton_kernels', True)  # IMPROVED: Pass Triton option
        },
        learning_rate=float(config['training']['learning_rate']),  # FIXED: Ensure float type
        batch_size=int(config['training']['batch_size']),
        accumulate_grad_batches=int(config['training']['accumulate_grad_batches']),
        max_sequence_length=int(sample_rate * config['data']['segment_seconds']),
        use_amp=bool(config['training']['mixed_precision']),
        use_checkpointing=bool(config['training']['gradient_checkpointing']),
        total_steps=total_steps,
        accelerator=accelerator  # IMPROVED: Pass accelerator to trainer
    )
    
    # IMPROVED: Initialize wandb if using Accelerate's wandb integration
    if accelerator.is_main_process and config.get('wandb', {}).get('enabled', True):
        wandb_config = {
            'project': config.get('wandb', {}).get('project', 'lycodec-v2-training'),
            'name': f"lycodec-v2-{config.get('seed', 42)}-{accelerator.num_processes}gpu",
            'config': {
                **config,
                'num_processes': accelerator.num_processes,
                'effective_batch_size': config['training']['batch_size'] * config['training']['accumulate_grad_batches'] * accelerator.num_processes,
                'gpu_type': 'V100-16GB',
                'total_steps': total_steps,
                'version': '2.0'
            },
            'tags': config.get('wandb', {}).get('tags', ['lycodec', 'f10c10', 'v100', f'{accelerator.num_processes}gpu', 'v2.0']),
            'notes': f"LyCodec v2.0 training on {accelerator.num_processes}x V100 16GB with Accelerate"
        }
        trainer.setup_wandb(wandb_config)
    
    # Setup validation data if requested
    val_dataloader = None
    if config.get('validation', {}).get('val_split', 0) > 0:
        val_dataloader = create_validation_loader(config, accelerator)
        if accelerator.is_main_process:
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
        if accelerator.is_main_process:
            print(f"Resumed from epoch {start_epoch}")
    
    # IMPROVED: Training epochs with Accelerate
    for epoch in range(start_epoch, config['training']['num_epochs']):
        # Train epoch
        avg_losses = trainer.train_epoch(dataloader, epoch)
        
        # Validation
        val_losses = {}
        if val_dataloader and (epoch + 1) % config.get('validation', {}).get('val_every', 10) == 0:
            val_losses = trainer.validate(val_dataloader)
            
            if accelerator.is_main_process:
                print("Validation results:")
                for key, value in val_losses.items():
                    print(f"  {key}: {value:.6f}")
        
        # Save checkpoint (only on main process)
        if accelerator.is_main_process:
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
        
        # IMPROVED: Use trainer's centralized logging
        trainer.log_epoch(epoch, avg_losses, val_losses, best_loss)
        
        # IMPROVED: Clear cache periodically to prevent memory buildup
        if torch.cuda.is_available() and (epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()
    
    # IMPROVED: Cleanup
    trainer.cleanup_wandb()
    if accelerator.is_main_process:
        print("Training completed successfully!")

if __name__ == '__main__':
    main()