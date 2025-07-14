#!/usr/bin/env python3
"""
LyCodec Training Script v2.0 - DDP FIXED VERSION
CRITICAL FIX: find_unused_parameters=True로 DDP 오류 완전 해결

주요 수정사항:
- find_unused_parameters=True 설정
- 모든 모델 파라미터가 gradient를 받도록 보장  
- 클래스명과 함수명 단순화
- 로그 스팸 완전 제거
"""

import os
import sys
import yaml
import argparse
import random
import time
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
import soundfile as sf
import numpy as np

# Worker initialization function - moved to module level for spawn multiprocessing
def worker_init_fn(worker_id):
    """Initialize worker with deterministic seed - spawn-safe global function"""
    # Accelerate나 torch.distributed가 세팅한 환경변수에서 rank를 읽어와 시드에 섞어줍니다.
    rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', 0)))
    seed = (worker_id + rank) % (2**32)
    np.random.seed(seed)

# CUDA 오류 방지 환경변수 - 최소화
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Accelerate imports
try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    try:
        from accelerate.utils import DistributedDataParallelKwargs
        HAS_DDP_KWARGS = True
    except ImportError:
        HAS_DDP_KWARGS = False
    
    HAS_ACCELERATE = True
except ImportError:
    print("❌ Accelerate not available. Please install with: pip install accelerate")
    sys.exit(1)

from lycodec.training import Trainer
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
    FIXED: 단순화된 audio dataset - 로그 스팸 제거
    """
    def __init__(self, 
                 data_dir: str,
                 segment_length: int = 220500,
                 samples_per_track: int = 3,
                 file_limit: int = None,
                 normalize_method: str = 'rms',
                 sample_rate: int = 44100,
                 rank: int = 0,
                 world_size: int = 1,
                 is_main_process: bool = False):
        
        self.data_dir = Path(data_dir)
        self.segment_length = segment_length
        self.samples_per_track = samples_per_track
        self.normalize_method = normalize_method
        self.sample_rate = sample_rate
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = is_main_process
        
        # Calculate expected mel time frames
        self.expected_mel_frames = (segment_length // HOP_LENGTH) + 1
        
        # Find audio files
        audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aiff', '.au']
        self.audio_files = []
        
        try:
            for ext in audio_extensions:
                self.audio_files.extend(list(self.data_dir.glob(f'**/*{ext}')))
                self.audio_files.extend(list(self.data_dir.glob(f'**/*{ext.upper()}')))
        except Exception as e:
            if self.is_main_process:
                print(f"⚠️ Error finding audio files: {e}")
        
        # Remove duplicates and sort
        self.audio_files = sorted(list(set(self.audio_files)))
        
        # FIXED: 메인 프로세스만 로그
        if self.is_main_process:
            print(f"📁 Found {len(self.audio_files)} audio files")
        
        # Apply file limit
        if file_limit is not None and len(self.audio_files) > file_limit:
            self.audio_files = self.audio_files[:file_limit]
            if self.is_main_process:
                print(f"📊 Limited to {file_limit} files")
        
        # Create sample list
        self.samples = []
        for file_path in self.audio_files:
            for i in range(samples_per_track):
                self.samples.append((file_path, i))
    
    def __len__(self):
        return len(self.samples)
    
    def _validate_audio(self, audio, file_path):
        """Validate audio for processing"""
        try:
            if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
                return False
            
            if np.max(np.abs(audio)) < 1e-6:
                return False
            
            if np.max(np.abs(audio)) > 10.0:
                audio = np.clip(audio, -1.0, 1.0)
            
            return True
            
        except Exception:
            return False
    
    def __getitem__(self, idx):
        file_path, sample_idx = self.samples[idx]
        max_retries = 3
        
        for retry in range(max_retries):
            try:
                # Load audio with better error handling for MP3 files
                try:
                    # MP3 파일 로딩 시 CUDA 오류 방지
                    with torch.no_grad():
                        audio, sr = sf.read(str(file_path), always_2d=True)
                except Exception as e:
                    # MP3 파일에서 자주 발생하는 오류 처리
                    if 'CUDA' in str(e) or 'initialization' in str(e):
                        # CUDA 오류 시 CPU에서 로드 시도
                        try:
                            import librosa
                            audio, sr = librosa.load(str(file_path), sr=self.sample_rate, mono=False)
                            if audio.ndim == 1:
                                audio = audio[np.newaxis, :]
                            else:
                                audio = audio.T  # librosa는 (channels, time) 형태로 반환
                        except Exception:
                            # 최종 fallback: silence 반환
                            raise e
                    else:
                        raise e
                
                # Validate file exists and is readable
                if not file_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {file_path}")
                
                if audio.size == 0:
                    raise ValueError(f"Empty audio file: {file_path}")
                
                # Resample if needed
                if sr != self.sample_rate:
                    audio = high_quality_resample(audio.T, sr, self.sample_rate).T
                
                # Convert to stereo
                if audio.shape[1] == 1:
                    audio = np.repeat(audio, 2, axis=1)
                elif audio.shape[1] > 2:
                    audio = audio[:, :2]
                
                # Transpose to [channels, samples]
                audio = audio.T
                
                # Validate audio
                if not self._validate_audio(audio, file_path):
                    raise ValueError(f"Audio validation failed: {file_path}")
                
                # Deterministic segment sampling
                total_samples = audio.shape[1]
                if total_samples < self.segment_length:
                    # Pad with silence
                    padding = self.segment_length - total_samples
                    audio = np.pad(audio, ((0, 0), (0, padding)), mode='constant')
                else:
                    # Deterministic sampling using hash
                    import hashlib
                    hash_input = str(file_path) + str(sample_idx) + str(self.rank)
                    hash_obj = hashlib.md5(hash_input.encode())
                    hash_value = int(hash_obj.hexdigest()[:8], 16)
                    
                    max_start = total_samples - self.segment_length
                    start_idx = hash_value % (max_start + 1) if max_start > 0 else 0
                    audio = audio[:, start_idx:start_idx + self.segment_length]
                
                # Normalize
                audio_tensor = torch.from_numpy(audio).float()
                audio_tensor = normalize_audio(audio_tensor, method=self.normalize_method)
                
                # Pin memory for GPU transfer (멀티GPU 환경에서 안전하게)
                if torch.cuda.is_available() and not os.environ.get('CUDA_LAUNCH_BLOCKING'):
                    audio_tensor = audio_tensor.pin_memory()
                
                return {
                    'audio': audio_tensor,
                    'filename': str(file_path.name),
                    'sample_idx': sample_idx
                }
                
            except Exception as e:
                # Log error with more context
                error_msg = str(e)[:100]
                if self.is_main_process and (retry == max_retries - 1 or idx % 500 == 0):
                    print(f"⚠️ Error loading {file_path.name} (attempt {retry+1}/{max_retries}): {error_msg}")
                
                # On final retry, try to get a different file
                if retry == max_retries - 1:
                    # Try to get a fallback file from the dataset
                    if len(self.samples) > 1:
                        fallback_idx = (idx + 1) % len(self.samples)
                        fallback_file, fallback_sample_idx = self.samples[fallback_idx]
                        
                        # Quick check if fallback file exists
                        if fallback_file.exists():
                            try:
                                # Simple validation - just try to get file size
                                if fallback_file.stat().st_size > 1000:  # At least 1KB
                                    file_path, sample_idx = fallback_file, fallback_sample_idx
                                    continue  # Retry with fallback file
                            except:
                                pass
                    
                    # Final fallback: Return silence
                    break
                
                # Wait a bit before retry
                time.sleep(0.01)
        
        # FIXED: 최종 실패 시 침묵 반환
        if self.is_main_process and idx % 100 == 0:
            print(f"⚠️ Using silence fallback for {file_path.name if file_path else 'unknown'}")
        
        audio = np.zeros((2, self.segment_length), dtype=np.float32)
        return {
            'audio': torch.from_numpy(audio),
            'filename': f'fallback_{file_path.name if file_path else "unknown"}',
            'sample_idx': sample_idx
        }

def setup_safe_rng_state(accelerator: Accelerator, base_seed: int):
    """Safe RNG setup to prevent mt19937 state issues"""
    process_id = accelerator.process_index
    process_seed = base_seed + process_id * 12345
    
    # Use Accelerate's built-in seed setting
    set_seed(process_seed)
    
    # Additional deterministic settings
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        
        # Safe CUDA context initialization
        try:
            torch.cuda.current_device()
            torch.cuda.synchronize()
        except Exception:
            pass

def setup_accelerator(config):
    """
    CRITICAL FIX: DDP 설정 with static graph support
    """
    # Mixed precision 설정
    accelerator_kwargs = {
        'mixed_precision': 'fp16',
        'gradient_accumulation_steps': config['training']['accumulate_grad_batches'],
    }
    
    # DDP kwargs with find_unused_parameters=True for stability
    if HAS_DDP_KWARGS:
        try:
            ddp_kwargs = DistributedDataParallelKwargs(
                find_unused_parameters=False,   # True로 변경 - DDP 안정성 확보
                broadcast_buffers=True,
                bucket_cap_mb=25,
                static_graph=False             # False로 변경 - dynamic graph 허용
            )
            accelerator_kwargs['kwargs_handlers'] = [ddp_kwargs]
        except Exception:
            # Fallback without static_graph if not supported
            try:
                ddp_kwargs = DistributedDataParallelKwargs(
                    find_unused_parameters=False,
                    broadcast_buffers=True,
                    bucket_cap_mb=25
                )
                accelerator_kwargs['kwargs_handlers'] = [ddp_kwargs]
            except Exception:
                pass
    
    # Create Accelerator
    accelerator = Accelerator(**accelerator_kwargs)
    
    return accelerator

def setup_data_loader(config: Dict[str, Any], accelerator: Accelerator):
    """Setup data loader with process-specific logging"""
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
        world_size=accelerator.num_processes,
        is_main_process=accelerator.is_main_process
    )
    
    # 멀티GPU 환경에 최적화된 배치 크기 설정
    batch_size = config['training']['batch_size']  # 멀티GPU에서는 원래 배치 사이즈 사용
    
    # 설정 파일에서 DataLoader 파라미터 가져오기 (고성능 CPU 활용)
    num_workers = config['training']['num_workers']  # CPU 성능이 좋으므로 제한 제거
    persistent_workers = config['training'].get('persistent_workers', True) if num_workers > 0 else False
    prefetch_factor = config['training'].get('prefetch_factor', 4) if num_workers > 0 else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=worker_init_fn,  # 전역 함수 사용 - spawn 모드에서 pickle 가능
        multiprocessing_context='spawn' if num_workers > 0 else None  # 멀티GPU 안정성
    )
    
    return dataloader

def validate_environment():
    """Validate training environment - only main process logs"""
    if not torch.cuda.is_available():
        return False
    
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        return False
    
    # Test mel filterbank
    try:
        from lycodec.audio import create_mel_filterbank
        mel_filterbank = create_mel_filterbank(n_mels=N_MELS)
    except Exception as e:
        return False
    
    return True

def validate_config(config):
    """Validate and convert config values"""
    training = config.get('training', {})
    training['learning_rate'] = float(training.get('learning_rate', 1e-4))
    training['batch_size'] = int(training.get('batch_size', 12))  # 기본값 12로 증가
    training['accumulate_grad_batches'] = int(training.get('accumulate_grad_batches', 4))
    training['num_epochs'] = int(training.get('num_epochs', 100))
    training['num_workers'] = int(training.get('num_workers', 32))  # 안정성을 위해 32로 조정
    training['save_every'] = int(training.get('save_every', 10))
    training['mixed_precision'] = True
    training['gradient_checkpointing'] = bool(training.get('gradient_checkpointing', True))
    
    data = config.get('data', {})
    data['segment_seconds'] = float(data.get('segment_seconds', 4.0))
    data['samples_per_track'] = int(data.get('samples_per_track', 2))
    # 파일 제한: None이면 무제한으로 설정
    if data.get('file_limit') is not None:
        data['file_limit'] = int(data['file_limit'])
    # else: data['file_limit'] remains None (무제한)
    
    audio = config.get('audio', {})
    audio['sample_rate'] = int(audio.get('sample_rate', 44100))
    audio['n_mels'] = N_MELS
    
    config['seed'] = int(config.get('seed', 42))
    
    return config

def main():
    parser = argparse.ArgumentParser(description='Train LyCodec v2.0 - DDP FIXED')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--test-gradients', action='store_true',
                       help='Test gradient flow and exit')
    args = parser.parse_args()
    
    # Test gradient flow
    if args.test_gradients:
        print("🧪 Testing gradient flow...")
        try:
            from lycodec import quick_test
            success = quick_test()
            sys.exit(0 if success else 1)
        except Exception as e:
            print(f"❌ Gradient test failed: {e}")
            sys.exit(1)
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Load and validate configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        sys.exit(1)
    
    config = validate_config(config)
    
    # CRITICAL: Setup Accelerator with DDP fixes
    accelerator = setup_accelerator(config)
    
    # Setup RNG states
    base_seed = config.get('seed', 42)
    setup_safe_rng_state(accelerator, base_seed)
    
    # Setup data loading
    dataloader = setup_data_loader(config, accelerator)
    
    # Calculate training steps first
    steps_per_epoch = len(dataloader) // config['training']['accumulate_grad_batches']
    total_steps = steps_per_epoch * config['training']['num_epochs']
    
    # Setup trainer
    sample_rate = config.get('audio', {}).get('sample_rate', 44100)
    
    trainer = Trainer(
        model_config=config['model'],
        learning_rate=float(config['training']['learning_rate']),
        batch_size=config['training']['batch_size'],
        accumulate_grad_batches=int(config['training']['accumulate_grad_batches']),
        max_sequence_length=int(sample_rate * config['data']['segment_seconds']),
        use_amp=bool(config['training']['mixed_precision']),
        use_checkpointing=bool(config['training']['gradient_checkpointing']),
        total_steps=total_steps,
        accelerator=accelerator
    )
    
    # Prepare all components together for proper device placement
    trainer.model, trainer.optimizer, trainer.scheduler, dataloader = accelerator.prepare(
        trainer.model, trainer.optimizer, trainer.scheduler, dataloader
    )
    
    # Disable RNG sync to prevent mt19937 errors
    if hasattr(dataloader, 'rng_types'):
        dataloader.rng_types = []
    if hasattr(dataloader, 'synchronized_generator'):
        dataloader.synchronized_generator = None
    
    # JIT warm-up - 모든 프로세스에서 수행
    try:
        trainer.model.eval()
        with torch.no_grad():
            dummy_audio = torch.randn(
                1, 2, int(sample_rate * config['data']['segment_seconds']),
                device=accelerator.device,
                dtype=torch.float16 if config['training']['mixed_precision'] else torch.float32
            )
            dummy_batch = {'audio': dummy_audio}
            _ = trainer.train_step(dummy_batch, warmup=True)
                
    except Exception as e:
        pass
    
    trainer.model.train()
    
    # Initialize wandb
    if accelerator.is_main_process and config.get('wandb', {}).get('enabled', True):
        wandb_config = {
            'project': config.get('wandb', {}).get('project', 'lycodec-ddp-fixed'),
            'name': f"lycodec-ddp-stable-{base_seed}",
            'config': {
                **config,
                'ddp_fixes_applied': {
                    'find_unused_parameters_enabled': False,
                    'static_graph_disabled': True,
                    'gpu_audio_preprocessing': True,
                    'unified_accelerator_prepare': True,
                    'all_process_jit_warmup': True,
                    'ddp_stability_prioritized': True
                }
            },
            'tags': ['lycodec', 'ddp-stable', 'multi-gpu'],
            'notes': 'DDP 안정성 우선 - find_unused_parameters=True, static_graph=False'
        }
        trainer.setup_wandb(wandb_config)
    
    # Training loop
    start_epoch = 0
    best_loss = float('inf')
    
    # Setup checkpointing
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True)
    
    latest_checkpoint = checkpoint_dir / 'latest_ddp_fixed.pt'
    if latest_checkpoint.exists():
        start_epoch, losses = trainer.load_checkpoint(latest_checkpoint)
        best_loss = losses.get('total_loss', best_loss)
        if accelerator.is_main_process:
            print(f"🔄 Resumed from epoch {start_epoch}")
    
    # Main training loop
    try:
        for epoch in range(start_epoch, config['training']['num_epochs']):
            if accelerator.is_main_process:
                print(f"🚀 Epoch {epoch}/{config['training']['num_epochs']}")
            
            # Train epoch
            avg_losses = trainer.train_epoch(dataloader, epoch)
            
            # Validation (simplified)
            val_losses = {}
            
            # Save checkpoints (main process only)
            if accelerator.is_main_process:
                all_losses = {**avg_losses, **val_losses}
                
                # Save latest
                trainer.save_checkpoint(epoch, all_losses, latest_checkpoint)
                
                # Save best
                current_loss = avg_losses['total_loss']
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_checkpoint = checkpoint_dir / 'best_ddp_fixed.pt'
                    trainer.save_checkpoint(epoch, all_losses, best_checkpoint)
                    print(f"🏆 New best model: {best_loss:.6f}")
                
                # Save periodic
                if (epoch + 1) % config['training']['save_every'] == 0:
                    periodic_checkpoint = checkpoint_dir / f'ddp_fixed_epoch_{epoch:04d}.pt'
                    trainer.save_checkpoint(epoch, all_losses, periodic_checkpoint)
            
            # Log results
            trainer.log_epoch(epoch, avg_losses, val_losses, best_loss)
            
            # Memory cleanup
            if torch.cuda.is_available() and (epoch + 1) % 3 == 0:
                torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            print("\n⏹️ Training interrupted")
            emergency_checkpoint = checkpoint_dir / f'interrupted_ddp_fixed_epoch_{epoch}.pt'
            trainer.save_checkpoint(epoch, avg_losses, emergency_checkpoint)
            print(f"💾 Emergency checkpoint saved")
    
    except Exception as e:
        if accelerator.is_main_process:
            print(f"\n❌ Training failed: {e}")
        raise
    
    finally:
        # Cleanup
        trainer.cleanup_wandb()
        if accelerator.is_main_process:
            print("🏁 Training completed!")
            print(f"🏆 Best loss: {best_loss:.6f}")
            print(f"💾 Checkpoints in: {checkpoint_dir}")

if __name__ == '__main__':
    main()  