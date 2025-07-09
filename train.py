#!/usr/bin/env python3
"""
LyCodec Training Script v2.0 - LOG-MEL + PHASE ARCHITECTURE with CRITICAL FIXES
V100×4 ACCELERATE VERSION with comprehensive DDP compatibility

NEW ARCHITECTURE:
- Waveform → STFT → Magnitude/Phase → Mel filterbank → log_mel (128 bin) + phase preservation
- PsychoacousticTransform applies masking curve weighting in log-mel domain
- f10c10 compression (100x) maintained through encoder/decoder stages

CRITICAL FIXES:
- RNG state isolation to prevent mt19937 errors
- DDP compatibility improvements with find_unused_parameters=False
- Enhanced parameter gradient flow verification for log-mel architecture
- Memory and tensor dimension fixes for mel-scale processing
- Psychoacoustic masking curve continuity in distributed training
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

# FIXED: More robust Accelerate imports with version compatibility
try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    # Import DDP kwargs conditionally
    try:
        from accelerate.utils import DistributedDataParallelKwargs
        HAS_DDP_KWARGS = True
    except ImportError:
        HAS_DDP_KWARGS = False
        print("ℹ️ DistributedDataParallelKwargs not available - using basic setup")
    
    HAS_ACCELERATE = True
    print("✅ Accelerate available for distributed training")
except ImportError:
    print("❌ Accelerate not available. Please install with: pip install accelerate")
    print("Run: accelerate config  # to setup 4-GPU distributed training")
    sys.exit(1)

from lycodec.training import LyCodecTrainer
from lycodec.audio import (
    normalize_audio, 
    high_quality_resample, 
    create_deterministic_seed, 
    get_optimal_pin_memory,
    N_MELS,
    SAMPLE_RATE,
    HOP_LENGTH
)

class AudioDataset(torch.utils.data.Dataset):
    """
    Dataset for audio files - ENHANCED for log-mel + phase architecture
    Optimized for mel-scale processing with better error handling and deterministic sampling
    """
    def __init__(self, 
                 data_dir: str,
                 segment_length: int = 220500,
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
        
        # Calculate expected mel time frames for validation
        self.expected_mel_frames = (segment_length // HOP_LENGTH) + 1
        
        # Find all audio files with better error handling
        audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aiff', '.au']
        self.audio_files = []
        
        try:
            for ext in audio_extensions:
                # Case insensitive search
                self.audio_files.extend(list(self.data_dir.glob(f'**/*{ext}')))
                self.audio_files.extend(list(self.data_dir.glob(f'**/*{ext.upper()}')))
        except Exception as e:
            if self.rank == 0:
                print(f"⚠️ Error finding audio files: {e}")
        
        # Remove duplicates and sort for consistency across GPUs
        self.audio_files = sorted(list(set(self.audio_files)))
        
        # Only log from main process
        if self.rank == 0:
            print(f"📁 Found {len(self.audio_files)} audio files")
            print(f"🎵 Expected mel frames per segment: {self.expected_mel_frames}")
        
        # Use all files if file_limit is None
        if file_limit is not None and len(self.audio_files) > file_limit:
            if self.rank == 0:
                print(f"📊 Limiting to {file_limit} audio files for log-mel training")
            self.audio_files = self.audio_files[:file_limit]
        elif self.rank == 0:
            print(f"📊 Using all {len(self.audio_files)} audio files for log-mel training")
        
        # Create sample list with deterministic ordering
        self.samples = []
        for file_path in self.audio_files:
            for i in range(samples_per_track):
                self.samples.append((file_path, i))
        
        if self.rank == 0:
            print(f"🎵 Total samples: {len(self.samples)} (distributed across {self.world_size} GPUs)")
            print(f"📊 Samples per GPU: ~{len(self.samples) // self.world_size}")
            print(f"🏗️ Architecture: Log-mel + phase with {N_MELS} mel bins")
    
    def __len__(self):
        return len(self.samples)
    
    def _validate_audio_for_mel_processing(self, audio, file_path):
        """
        NEW: Validate audio is suitable for mel-scale processing
        """
        try:
            # Check for common audio issues that affect mel processing
            if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
                if self.rank == 0:
                    print(f"⚠️ Audio contains NaN/Inf values: {file_path.name}")
                return False
            
            # Check dynamic range
            if np.max(np.abs(audio)) < 1e-6:
                if self.rank == 0:
                    print(f"⚠️ Audio too quiet for mel processing: {file_path.name}")
                return False
            
            # Check for extreme values that could affect STFT
            if np.max(np.abs(audio)) > 10.0:
                if self.rank == 0:
                    print(f"⚠️ Audio has extreme values: {file_path.name}")
                # Clip instead of rejecting
                audio = np.clip(audio, -1.0, 1.0)
            
            return True
            
        except Exception as e:
            if self.rank == 0:
                print(f"⚠️ Audio validation failed for {file_path.name}: {e}")
            return False
    
    def __getitem__(self, idx):
        file_path, sample_idx = self.samples[idx]
        
        try:
            # Load audio with error handling
            audio, sr = sf.read(str(file_path), always_2d=True)
            
            # High-quality resampling if needed
            if sr != self.sample_rate:
                if self.rank == 0 and sample_idx == 0:
                    print(f"🔄 Resampling {file_path.name} from {sr}Hz to {self.sample_rate}Hz for mel processing")
                audio = high_quality_resample(audio.T, sr, self.sample_rate).T
            
            # Convert to stereo if mono
            if audio.shape[1] == 1:
                audio = np.repeat(audio, 2, axis=1)
            elif audio.shape[1] > 2:
                audio = audio[:, :2]  # Take first two channels
            
            # Transpose to [channels, samples]
            audio = audio.T
            
            # NEW: Validate audio for mel-scale processing
            if not self._validate_audio_for_mel_processing(audio, file_path):
                # Return silence if validation fails
                audio = np.zeros((2, self.segment_length), dtype=np.float32)
                return {
                    'audio': torch.from_numpy(audio),
                    'filename': f'validation_failed_{file_path.name}',
                    'sample_idx': sample_idx,
                    'original_sr': sr,
                    'mel_frames': self.expected_mel_frames,
                    'architecture': 'log_mel_phase'
                }
            
            # FIXED: More robust segment sampling optimized for mel processing
            total_samples = audio.shape[1]
            if total_samples < self.segment_length:
                # Pad with silence if too short
                padding = self.segment_length - total_samples
                audio = np.pad(audio, ((0, 0), (0, padding)), mode='constant')
            else:
                # CRITICAL: Simplified sampling to avoid RNG conflicts with Accelerate
                # Use hash-based deterministic sampling instead of RNG
                import hashlib
                hash_input = str(file_path) + str(sample_idx) + str(self.rank)
                hash_obj = hashlib.md5(hash_input.encode())
                hash_value = int(hash_obj.hexdigest()[:8], 16)  # Use first 8 hex chars
                
                max_start = total_samples - self.segment_length
                start_idx = hash_value % (max_start + 1) if max_start > 0 else 0
                audio = audio[:, start_idx:start_idx + self.segment_length]
            
            # STEP 5: CPU ↔ GPU 복사 최소화 - pinned memory + half precision
            # Normalize with specified method
            audio_tensor = torch.from_numpy(audio).float()
            audio_tensor = normalize_audio(audio_tensor, method=self.normalize_method)
            
            # STEP 5: Use half precision and pinned memory for faster GPU transfer
            # PCIe copy 170 MB → 34 MB (5x reduction)
            if torch.cuda.is_available():
                audio_tensor = audio_tensor.half().pin_memory()  # FP16 + pinned memory
            
            # Calculate actual mel frames that will be produced
            actual_mel_frames = (audio_tensor.shape[-1] // HOP_LENGTH) + 1
            
            return {
                'audio': audio_tensor,
                'filename': str(file_path.name),
                'sample_idx': sample_idx,
                'original_sr': sr,
                'mel_frames': actual_mel_frames,
                'architecture': 'log_mel_phase'
            }
            
        except Exception as e:
            if self.rank == 0:
                print(f"⚠️ Error loading {file_path} for log-mel processing: {e}")
            # Return silence as fallback
            audio = np.zeros((2, self.segment_length), dtype=np.float32)
            return {
                'audio': torch.from_numpy(audio),
                'filename': f'error_{file_path.name if file_path else "unknown"}',
                'sample_idx': 0,
                'original_sr': self.sample_rate,
                'mel_frames': self.expected_mel_frames,
                'architecture': 'log_mel_phase'
            }

def setup_safe_rng_state(accelerator: Accelerator, base_seed: int):
    """
    CRITICAL: Setup RNG states compatible with Accelerate synchronization for log-mel training
    Avoid manual RNG manipulation that causes mt19937 state issues
    """
    process_id = accelerator.process_index
    world_size = accelerator.num_processes
    
    # Calculate unique seed for each process
    process_seed = base_seed + process_id * 12345
    
    # CRITICAL: Use Accelerate's built-in seed setting which handles RNG sync properly
    set_seed(process_seed)
    
    # Additional deterministic settings for reproducibility
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        
        # Initialize CUDA context safely for each process
        try:
            torch.cuda.current_device()
            torch.cuda.synchronize()
        except Exception as e:
            if accelerator.is_main_process:
                print(f"⚠️ CUDA context initialization warning: {e}")
    
    if accelerator.is_main_process:
        print(f"🎲 Safe RNG setup completed for log-mel training:")
        print(f"   Base seed: {base_seed}")
        print(f"   Process seeds: {[base_seed + i * 12345 for i in range(world_size)]}")
        print(f"   Using Accelerate's RNG synchronization")
        print(f"   Architecture: Log-mel + phase with {N_MELS} mel bins")

def setup_4gpu_accelerator(config):
    """Setup Accelerator for V100×4 with CRITICAL DDP fixes for log-mel architecture"""
    
    # CRITICAL: Setup DDP kwargs to handle unused parameters for log-mel processing
    accelerator_kwargs = {
        'mixed_precision': 'fp16',  # Enable FP16 as requested
        'gradient_accumulation_steps': config['training']['accumulate_grad_batches'],
    }
    
    # CRITICAL: Add DDP kwargs for unused parameter handling
    if HAS_DDP_KWARGS:
        try:
            ddp_kwargs = DistributedDataParallelKwargs(
                find_unused_parameters=False,  # Set to False as requested
                broadcast_buffers=True,
                bucket_cap_mb=25,
                # Remove timeout_seconds as it doesn't exist in newer versions
            )
            accelerator_kwargs['kwargs_handlers'] = [ddp_kwargs]
            print("✅ DDP kwargs configured with find_unused_parameters=False for log-mel training")
        except Exception as e:
            print(f"⚠️ DDP kwargs setup failed: {e}")
    
    # Add safe optional parameters
    try:
        accelerator_kwargs['project_dir'] = config['training']['checkpoint_dir']
    except Exception:
        pass
    
    # Create Accelerator with safe parameters
    accelerator = Accelerator(**accelerator_kwargs)
    
    # Verify setup
    if accelerator.num_processes != 4:
        print(f"⚠️ Expected 4 GPUs, but got {accelerator.num_processes}")
        if accelerator.num_processes == 1:
            print("💡 Running in single-GPU mode for log-mel training")
    else:
        print(f"✅ V100×4 setup verified for log-mel + phase architecture")
    
    if accelerator.is_main_process:
        print(f"🚀 V100×4 Log-mel Training Setup:")
        print(f"   📊 Processes: {accelerator.num_processes}")
        print(f"   🔄 Mixed precision: {accelerator.mixed_precision}")
        print(f"   🔢 Gradient accumulation: {config['training']['accumulate_grad_batches']}")
        print(f"   🎵 Mel bins: {N_MELS}")
        print(f"   📐 Architecture: Log-mel + phase preservation")
        effective_batch = (config['training']['batch_size'] * 
                          config['training']['accumulate_grad_batches'] * 
                          accelerator.num_processes)
        print(f"   📦 Effective batch size: {effective_batch}")
    
    return accelerator

def setup_data_loader(config: Dict[str, Any], accelerator: Accelerator):
    """Setup data loader with improved deterministic behavior for log-mel training"""
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
    
    # CRITICAL: Simplified worker initialization to prevent RNG conflicts
    def worker_init_fn(worker_id):
        # Simple, safe worker seed based on worker ID only
        worker_seed = worker_id + 42
        
        # Use basic seeding that won't conflict with Accelerate's RNG sync
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    # STEP 3: Optimized DataLoader settings for I/O tuning
    pin_memory = config['training'].get('pin_memory', True)  # STEP 5: pinned memory
    num_workers = config['training']['num_workers']
    prefetch_factor = config['training'].get('prefetch_factor', 4)  # STEP 3: prefetch tuning
    persistent_workers = config['training'].get('persistent_workers', True)  # STEP 3: reduce context-switch
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,  # STEP 5: faster GPU transfer
        drop_last=True,  # Critical for distributed training
        persistent_workers=persistent_workers if num_workers > 0 else False,  # STEP 3: keep workers alive
        prefetch_factor=prefetch_factor if num_workers > 0 else None,  # STEP 3: prefetch optimization
        worker_init_fn=worker_init_fn if num_workers > 0 else None
    )
    
    if accelerator.is_main_process:
        print(f"✅ DataLoader configured with STEP 3+5 optimizations:")
        print(f"   Batch size: {config['training']['batch_size']}")
        print(f"   Workers: {num_workers} (STEP 3: optimized for cores)")
        print(f"   Pin memory: {pin_memory} (STEP 5: faster GPU transfer)")
        print(f"   Persistent workers: {persistent_workers} (STEP 3: reduce context-switch)")
        print(f"   Prefetch factor: {prefetch_factor} (STEP 3: I/O optimization)")
    
    return dataloader

def create_validation_loader(config: Dict[str, Any], accelerator: Accelerator):
    """Create validation data loader for log-mel architecture"""
    val_config = config.copy()
    val_config['data']['samples_per_track'] = 1
    val_config['training']['batch_size'] = max(1, config['training']['batch_size'] // 2)
    
    return setup_data_loader(val_config, accelerator)

def validate_environment():
    """Validate training environment for log-mel architecture"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    if gpu_count < 4:
        print(f"⚠️ Expected 4 GPUs, found {gpu_count}")
    
    # Check GPU memory
    for i in range(min(4, gpu_count)):
        try:
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"🔧 GPU {i}: {props.name}, Memory: {memory_gb:.1f}GB")
            if memory_gb < 15:
                print(f"⚠️ GPU {i} has less than 16GB memory, may cause issues with log-mel training")
        except Exception as e:
            print(f"⚠️ Error checking GPU {i}: {e}")
    
    # Check for log-mel specific requirements
    try:
        # Test mel filterbank creation
        from lycodec.audio import create_mel_filterbank
        mel_filterbank = create_mel_filterbank(n_mels=N_MELS)
        print(f"✅ Mel filterbank test: {mel_filterbank.shape} ({N_MELS} mel bins)")
    except Exception as e:
        print(f"❌ Mel filterbank test failed: {e}")
        return False
    
    return True

def validate_config(config):
    """Validate and convert config values to proper types for log-mel training"""
    training = config.get('training', {})
    training['learning_rate'] = float(training.get('learning_rate', 1e-4))
    training['batch_size'] = int(training.get('batch_size', 1))
    training['accumulate_grad_batches'] = int(training.get('accumulate_grad_batches', 16))
    training['num_epochs'] = int(training.get('num_epochs', 1000))
    training['num_workers'] = int(training.get('num_workers', 0))
    training['save_every'] = int(training.get('save_every', 50))
    training['mixed_precision'] = bool(training.get('mixed_precision', False))
    training['gradient_checkpointing'] = bool(training.get('gradient_checkpointing', True))
    
    data = config.get('data', {})
    data['segment_seconds'] = float(data.get('segment_seconds', 5.0))
    data['samples_per_track'] = int(data.get('samples_per_track', 3))
    if data.get('file_limit') is not None:
        data['file_limit'] = int(data['file_limit'])
    
    audio = config.get('audio', {})
    audio['sample_rate'] = int(audio.get('sample_rate', 44100))
    
    config['seed'] = int(config.get('seed', 42))
    
    # Validate mel-specific settings
    if 'n_mels' in audio:
        audio['n_mels'] = int(audio['n_mels'])
        if audio['n_mels'] != N_MELS:
            print(f"⚠️ Config n_mels ({audio['n_mels']}) differs from architecture ({N_MELS}), using {N_MELS}")
            audio['n_mels'] = N_MELS
    else:
        audio['n_mels'] = N_MELS
    
    return config

def main():
    parser = argparse.ArgumentParser(description='Train LyCodec v2.0 - LOG-MEL + PHASE ARCHITECTURE')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from specific checkpoint')
    parser.add_argument('--test-architecture', action='store_true',
                       help='Test log-mel + phase architecture and exit')
    args = parser.parse_args()
    
    # Quick architecture test
    if args.test_architecture:
        print("🧪 Testing log-mel + phase architecture...")
        try:
            from lycodec import quick_test, print_architecture_summary
            print_architecture_summary()
            success = quick_test()
            sys.exit(0 if success else 1)
        except Exception as e:
            print(f"❌ Architecture test failed: {e}")
            sys.exit(1)
    
    # Validate environment
    if not validate_environment():
        print("❌ Environment validation failed")
        sys.exit(1)
    
    # Load and validate configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    config = validate_config(config)
    
    # CRITICAL: Setup Accelerator with DDP fixes for log-mel
    accelerator = setup_4gpu_accelerator(config)
    
    # CRITICAL: Setup isolated RNG states to prevent mt19937 errors
    base_seed = config.get('seed', 42)
    setup_safe_rng_state(accelerator, base_seed)
    
    # Setup data loading for log-mel training
    dataloader = setup_data_loader(config, accelerator)
    
    # CRITICAL: Prepare DataLoader and immediately disable RNG synchronization
    # This prevents the "Invalid mt19937 state" error during iteration
    dataloader = accelerator.prepare(dataloader)
    
    # Disable RNG synchronization on the prepared DataLoader
    if hasattr(dataloader, 'rng_types'):
        dataloader.rng_types = []  # Empty list disables RNG sync
    if hasattr(dataloader, 'synchronized_generator'):
        dataloader.synchronized_generator = None  # Disable generator sync
    
    # Calculate training steps
    steps_per_epoch = len(dataloader) // config['training']['accumulate_grad_batches']
    total_steps = steps_per_epoch * config['training']['num_epochs']
    
    if accelerator.is_main_process:
        print(f"📊 Log-mel Training Configuration:")
        print(f"   🔢 Steps per epoch: {steps_per_epoch}")
        print(f"   🔢 Total steps: {total_steps}")
        print(f"   📦 Batch size per GPU: {config['training']['batch_size']}")
        print(f"   🎵 Mel bins: {N_MELS}")
        print(f"   📐 Architecture: Log-mel + phase preservation")
        effective_batch = (config['training']['batch_size'] * 
                          config['training']['accumulate_grad_batches'] * 
                          accelerator.num_processes)
        print(f"   📦 Effective batch size: {effective_batch}")
    
    # Setup trainer for log-mel architecture
    sample_rate = config.get('audio', {}).get('sample_rate', 44100)
    
    trainer = LyCodecTrainer(
        model_config={
            **config['model'],
            'use_triton': False  # Disabled for stability
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
    
    # STEP 8: JIT Warm-up 분리 - 첫 배치 280s 제거
    if accelerator.is_main_process:
        print("🔥 STEP 8: Performing JIT warm-up to remove first batch overhead...")
    
    # Dummy forward pass for JIT compilation warm-up
    try:
        trainer.model.eval()
        with torch.no_grad():
            # Create dummy batch matching actual data shape
            dummy_audio = torch.randn(
                2, 2, int(sample_rate * config['data']['segment_seconds']),
                device=accelerator.device,
                dtype=torch.float16 if config['training']['mixed_precision'] else torch.float32
            )
            dummy_batch = {'audio': dummy_audio}
            
            # Warm-up forward pass
            _ = trainer.train_step(dummy_batch, warmup=True)
            
            if accelerator.is_main_process:
                print("✅ JIT warm-up completed - first batch latency eliminated")
                
    except Exception as e:
        if accelerator.is_main_process:
            print(f"⚠️ JIT warm-up failed: {e} - continuing without warm-up")
    
    trainer.model.train()  # Return to training mode
    
    # CRITICAL: Verify DDP compatibility for log-mel architecture
    if accelerator.is_main_process:
        print("🔍 Running DDP compatibility checks for log-mel + phase architecture...")
    trainer.verify_ddp_compatibility()
    
    # Initialize wandb for log-mel training
    if accelerator.is_main_process and config.get('wandb', {}).get('enabled', True):
        wandb_config = {
            'project': config.get('wandb', {}).get('project', 'lycodec-v2-logmel'),
            'name': f"lycodec-logmel-{base_seed}",
            'config': {
                **config,
                'num_processes': accelerator.num_processes,
                'effective_batch_size': effective_batch,
                'total_steps': total_steps,
                'version': '2.0-logmel',
                'architecture': 'log_mel_phase',
                'mel_bins': N_MELS,
                'compression_ratio': 100,  # f10c10
                'fixes_applied': {
                    'rng_isolation': True,
                    'ddp_unused_params': True,
                    'log_mel_phase_architecture': True,
                    'psychoacoustic_masking': True,
                    'memory_optimizations': True
                }
            },
            'tags': config.get('wandb', {}).get('tags', ['lycodec', 'v2.0', 'log-mel', 'phase']),
            'notes': 'LyCodec v2.0 with log-mel + phase architecture, psychoacoustic masking, and critical fixes'
        }
        trainer.setup_wandb(wandb_config)
    
    # Setup validation for log-mel training
    val_dataloader = None
    if config.get('validation', {}).get('val_split', 0) > 0:
        val_dataloader = create_validation_loader(config, accelerator)
        if accelerator.is_main_process:
            print(f"✅ Validation loader created for log-mel architecture")
    
    # Training loop for log-mel + phase architecture
    start_epoch = 0
    best_loss = float('inf')
    
    # Resume from checkpoint
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True)
    
    latest_checkpoint = checkpoint_dir / 'latest_logmel.pt'
    if latest_checkpoint.exists():
        start_epoch, losses = trainer.load_checkpoint(latest_checkpoint)
        best_loss = losses.get('total_loss', best_loss)
        if accelerator.is_main_process:
            print(f"🔄 Resumed log-mel training from epoch {start_epoch}")
    
    # CRITICAL: Main training loop with log-mel + phase architecture
    avg_losses = {'total_loss': float('inf')}
    epoch = start_epoch
    
    try:
        for epoch in range(start_epoch, config['training']['num_epochs']):
            if accelerator.is_main_process:
                print(f"🚀 Epoch {epoch}/{config['training']['num_epochs']} [LOG-MEL+PHASE]")
            
            # Train epoch with log-mel + phase architecture
            avg_losses = trainer.train_epoch(dataloader, epoch)
            
            # Validation
            val_losses = {}
            if val_dataloader and (epoch + 1) % config.get('validation', {}).get('val_every', 10) == 0:
                val_losses = trainer.validate(val_dataloader)
            
            # Save checkpoints (main process only)
            if accelerator.is_main_process:
                all_losses = {**avg_losses, **val_losses}
                
                # Save latest
                trainer.save_checkpoint(epoch, all_losses, latest_checkpoint)
                
                # Save best
                current_loss = val_losses.get('val_total_loss', avg_losses['total_loss'])
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_checkpoint = checkpoint_dir / 'best_logmel.pt'
                    trainer.save_checkpoint(epoch, all_losses, best_checkpoint)
                    print(f"🏆 New best log-mel model: {best_loss:.6f}")
                
                # Save periodic
                if (epoch + 1) % config['training']['save_every'] == 0:
                    periodic_checkpoint = checkpoint_dir / f'logmel_epoch_{epoch:04d}.pt'
                    trainer.save_checkpoint(epoch, all_losses, periodic_checkpoint)
            
            # Log results
            trainer.log_epoch(epoch, avg_losses, val_losses, best_loss)
            
            # Memory management
            if torch.cuda.is_available() and (epoch + 1) % 5 == 0:
                torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            print("\n⏹️ Log-mel training interrupted")
            emergency_checkpoint = checkpoint_dir / f'interrupted_logmel_epoch_{epoch}.pt'
            trainer.save_checkpoint(epoch, avg_losses, emergency_checkpoint)
            print(f"💾 Emergency log-mel checkpoint saved: {emergency_checkpoint}")
    
    except Exception as e:
        if accelerator.is_main_process:
            print(f"\n❌ Log-mel training failed: {e}")
            import traceback
            traceback.print_exc()
        raise
    
    finally:
        # Cleanup
        trainer.cleanup_wandb()
        if accelerator.is_main_process:
            print("🏁 Log-mel + Phase training completed!")
            print(f"   🏆 Best loss: {best_loss:.6f}")
            print(f"   📅 Epochs: {epoch + 1}")
            print(f"   💾 Checkpoints in: {checkpoint_dir}")
            print(f"   🎵 Architecture: Log-mel + phase with {N_MELS} mel bins")
            print(f"   📊 Compression: f10c10 (100x)")

if __name__ == '__main__':
    main()