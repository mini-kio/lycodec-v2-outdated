import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import os
import time
import logging
from pathlib import Path
from tqdm import tqdm

# Use Accelerate for V100×4 distributed training
try:
    from accelerate import Accelerator
    HAS_ACCELERATE = True
    print("✅ Accelerate available for V100×4 distributed training")
except ImportError:
    HAS_ACCELERATE = False
    print("❌ Accelerate not available - please install: pip install accelerate")
    
    # Dummy Accelerator for fallback
    class DummyAccelerator:
        def __init__(self):
            self.is_main_process = True
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.num_processes = 1
            self.mixed_precision = 'no'
        
        def prepare(self, *args):
            if len(args) == 1:
                return args[0]
            return args
        
        def backward(self, loss):
            loss.backward()
        
        def accumulate(self, model):
            from contextlib import nullcontext
            return nullcontext()
        
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

from .models import LyCodecModel
from .audio import (
    SpectralLoss, 
    to_complex_spec, 
    to_magnitude_phase, 
    to_waveform
)

# Import wandb in training module
try:
    import wandb
    HAS_WANDB = True
    print("✅ WandB available for experiment tracking")
except ImportError:
    HAS_WANDB = False
    print("ℹ️ WandB not available - training metrics won't be logged")

class LyCodecTrainer:
    """
    LyCodec trainer optimized for V100×4 16GB setup - STABLE VERSION
    Features:
    - Accelerate for easy distributed training on 4 GPUs
    - Memory-efficient batching optimized for 16GB VRAM
    - Stable PyTorch implementation (Triton disabled)
    - Progress tracking with tqdm
    - Enhanced error handling and recovery
    """
    
    def __init__(self, 
                 model_config=None,
                 learning_rate=1e-4,
                 batch_size=4,  # Per-GPU batch size for V100 16GB
                 accumulate_grad_batches=4,  # Effective batch size: 64 (4 GPUs × 4 batch × 4 accum)
                 max_sequence_length=220500,  # 5 seconds at 44.1kHz
                 use_amp=True,
                 use_checkpointing=True,
                 total_steps=None,
                 accelerator: Accelerator = None):
        
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.accumulate_grad_batches = int(accumulate_grad_batches)
        self.max_sequence_length = int(max_sequence_length)
        self.use_amp = bool(use_amp)
        self.use_checkpointing = bool(use_checkpointing)
        self.total_steps = total_steps
        
        # Use provided accelerator or create dummy
        self.accelerator = accelerator or DummyAccelerator()
        self.is_main_process = self.accelerator.is_main_process
        
        # Setup logging for V100×4
        self._setup_logging()
        
        # Initialize model with stable configuration
        model_config = model_config or {}
        # FIXED: Force disable Triton for V100×4 stability
        model_config['use_triton'] = False
        
        self.model = LyCodecModel(**model_config)
        
        if self.is_main_process:
            self.logger.info(f"Model initialized with config: {model_config}")
        
        # Enable gradient checkpointing for memory efficiency on 16GB cards
        if use_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Loss functions with stable implementation
        self.spectral_loss = SpectralLoss(n_ffts=[512, 1024, 2048], alpha=1.0, beta=0.1, use_triton=False)
        self.mse_loss = nn.MSELoss()
        
        # Optimizer optimized for V100×4
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.learning_rate),
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-6
        )
        
        # Learning rate scheduler with warm restart for long training
        self.scheduler = None
        self._create_scheduler()
        
        # Prepare model, optimizer, scheduler with Accelerate for 4-GPU distributed training
        if self.accelerator and HAS_ACCELERATE:
            self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler
            )
            
            if self.is_main_process:
                self.logger.info(f"V100×4 setup: {self.accelerator.num_processes} GPUs, mixed precision: {self.accelerator.mixed_precision}")
        
        # Initialize wandb tracking
        self.wandb_run = None
        
        # V100 specific memory tracking
        self.gpu_memory_threshold = 14.0  # 14GB threshold for 16GB V100s
    
    def _setup_logging(self):
        """Setup logging only for main process"""
        if self.is_main_process:
            from logging.handlers import RotatingFileHandler
            
            # Setup rotating file handler for V100×4 training logs
            file_handler = RotatingFileHandler(
                'v100x4_training.log',
                maxBytes=20*1024*1024,  # 20MB max per file
                backupCount=10  # Keep 10 backup files
            )
            
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - [GPU:%(process)d] - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    file_handler
                ]
            )
            self.logger = logging.getLogger(__name__)
            self.logger.info("🚀 V100×4 training logger initialized")
        else:
            # Null logger for non-main processes
            self.logger = logging.getLogger(__name__)
            self.logger.addHandler(logging.NullHandler())
            self.logger.setLevel(logging.CRITICAL)
    
    def _create_scheduler(self):
        """Create scheduler optimized for long V100×4 training"""
        total_steps = self.total_steps or 100000
        
        # Larger T_0 for V100×4 distributed training
        T_0 = max(total_steps // 8, 2000)  # Larger restart intervals
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=T_0,
            T_mult=2,
            eta_min=float(self.learning_rate) / 100,
            last_epoch=-1
        )
    
    def update_total_steps(self, total_steps: int):
        """Update scheduler with correct total steps after knowing dataset size"""
        self.total_steps = total_steps
        self._create_scheduler()
        
        # Re-prepare scheduler with Accelerate
        if self.accelerator and HAS_ACCELERATE:
            self.scheduler = self.accelerator.prepare(self.scheduler)
        
        if self.is_main_process:
            self.logger.info(f"Updated scheduler for V100×4 training: total_steps={total_steps}")
        
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing with safer implementation for V100×4"""
        try:
            from torch.utils.checkpoint import checkpoint
            
            def create_checkpointed_forward(original_forward, module_name="unknown"):
                """Create a checkpointed version with V100 optimization"""
                def checkpointed_forward(*args, **kwargs):
                    try:
                        # Use non-reentrant checkpointing for V100 stability
                        return checkpoint(
                            original_forward, 
                            *args, 
                            use_reentrant=False,
                            **kwargs
                        )
                    except Exception as e:
                        # Fallback to original forward on any error
                        if self.is_main_process:
                            self.logger.warning(f"Checkpointing failed for {module_name}: {e}")
                        return original_forward(*args, **kwargs)
                return checkpointed_forward
            
            # Track patched modules to avoid double-patching
            if not hasattr(self, '_checkpointed_modules'):
                self._checkpointed_modules = set()
            
            def apply_checkpointing_to_module(module, module_path=""):
                """Apply checkpointing to ResidualBlock modules"""
                module_id = id(module)
                
                if module_id in self._checkpointed_modules:
                    return
                
                module_class_name = module.__class__.__name__
                if 'ResidualBlock' in module_class_name and hasattr(module, 'forward'):
                    try:
                        if not hasattr(module, '_original_forward'):
                            module._original_forward = module.forward
                            module.forward = create_checkpointed_forward(
                                module._original_forward, 
                                f"{module_path}.{module_class_name}"
                            )
                            module._ckpt_patched = True
                            self._checkpointed_modules.add(module_id)
                            
                            if self.is_main_process:
                                self.logger.debug(f"Applied checkpointing to {module_path}")
                    
                    except Exception as e:
                        if self.is_main_process:
                            self.logger.warning(f"Failed to apply checkpointing to {module_path}: {e}")
            
            # Apply to encoder and decoder ResidualBlocks
            patched_count = 0
            
            try:
                if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layers'):
                    for i, layer in enumerate(self.model.encoder.layers):
                        if hasattr(layer, 'psych_attn'):  # ResidualBlock identifier
                            apply_checkpointing_to_module(layer, f"encoder.layers[{i}]")
                            patched_count += 1
            except Exception as e:
                if self.is_main_process:
                    self.logger.warning(f"Error applying checkpointing to encoder: {e}")
            
            try:
                if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'layers'):
                    for i, layer in enumerate(self.model.decoder.layers):
                        if hasattr(layer, 'psych_attn'):  # ResidualBlock identifier
                            apply_checkpointing_to_module(layer, f"decoder.layers[{i}]")
                            patched_count += 1
            except Exception as e:
                if self.is_main_process:
                    self.logger.warning(f"Error applying checkpointing to decoder: {e}")
            
            if self.is_main_process:
                if patched_count > 0:
                    self.logger.info(f"✅ Applied V100 optimized checkpointing to {patched_count} ResidualBlocks")
                else:
                    self.logger.warning("No ResidualBlocks found for checkpointing")
            
        except ImportError as e:
            self.logger.error(f"Could not import checkpoint: {e}")
            self.use_checkpointing = False
        except Exception as e:
            if self.is_main_process:
                self.logger.warning(f"Could not apply gradient checkpointing: {e}")
            self.use_checkpointing = False
    
    def setup_wandb(self, wandb_config=None):
        """Setup wandb tracking for V100×4 experiments"""
        if self.is_main_process and HAS_WANDB and wandb_config:
            try:
                self.wandb_run = wandb.init(
                    project=wandb_config['project'],
                    name=wandb_config['name'],
                    config=wandb_config['config'],
                    tags=wandb_config['tags'],
                    notes=wandb_config['notes']
                )
                self.logger.info(f"🎯 WandB initialized for V100×4: {self.wandb_run.name}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize wandb: {e}")
                self.wandb_run = None
    
    def cleanup_wandb(self):
        """Cleanup wandb run"""
        if self.wandb_run is not None:
            wandb.finish()
            self.wandb_run = None
            if self.is_main_process:
                self.logger.info("WandB run finished")
    
    def _monitor_gpu_memory(self):
        """Monitor GPU memory usage for V100 16GB cards"""
        if torch.cuda.is_available():
            try:
                for gpu_id in range(torch.cuda.device_count()):
                    memory_used = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                    memory_cached = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                    
                    if memory_used > self.gpu_memory_threshold:
                        if self.is_main_process:
                            self.logger.warning(f"🚨 GPU {gpu_id} memory usage high: {memory_used:.1f}GB / 16GB")
                        # Emergency memory cleanup
                        torch.cuda.empty_cache()
            except Exception as e:
                if self.is_main_process:
                    self.logger.debug(f"Memory monitoring failed: {e}")
    
    def compute_loss(self, pred_real, pred_imag, target_real, target_imag, target_audio, pred_latent=None):
        """Compute multi-component loss with improved numerical stability"""
        device = pred_real.device
        
        # Reconstruct complex spectrogram
        pred_complex = torch.complex(pred_real, pred_imag)
        target_complex = torch.complex(target_real, target_imag)
        
        # Magnitude and phase losses with better numerical stability
        pred_mag = torch.abs(pred_complex)
        target_mag = torch.abs(target_complex)
        magnitude_loss = F.l1_loss(pred_mag, target_mag)
        
        # Phase loss with magnitude weighting - improved stability
        magnitude_weight = target_mag / (target_mag.amax(dim=(-1, -2, -3), keepdim=True) + 1e-8)
        pred_phase = torch.angle(pred_complex)
        target_phase = torch.angle(target_complex)
        
        # Use 1-cos for phase loss with clipping for stability
        phase_diff_cos = torch.cos(pred_phase - target_phase)
        phase_loss = F.mse_loss(
            torch.clamp((1 - phase_diff_cos) * magnitude_weight, 0, 2), 
            torch.zeros_like(phase_diff_cos)
        )
        
        # Audio reconstruction losses with better error handling
        try:
            pred_audio = to_waveform(pred_complex)
            
            # Ensure same length
            min_len = min(pred_audio.shape[-1], target_audio.shape[-1])
            pred_audio = pred_audio[..., :min_len]
            target_audio_trimmed = target_audio[..., :min_len]
            
            # Multi-scale spectral loss
            spectral_loss = self.spectral_loss(pred_audio, target_audio_trimmed)
            
            # Time-domain loss with gradient clipping for stability
            time_loss = F.l1_loss(pred_audio, target_audio_trimmed)
            
        except Exception as e:
            if self.is_main_process:
                self.logger.warning(f"Audio reconstruction failed: {e}")
            # Return zero losses with gradients for training continuity
            spectral_loss = torch.tensor(0.0, device=device, requires_grad=True)
            time_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Latent regularization
        latent_loss = torch.tensor(0.0, device=device)
        if pred_latent is not None:
            latent_loss = torch.mean(torch.abs(pred_latent))
        
        # Combine losses with V100 optimized weights
        total_loss = (
            1.0 * magnitude_loss +
            0.1 * phase_loss +
            0.5 * spectral_loss +
            0.3 * time_loss +
            0.01 * latent_loss
        )
        
        return {
            'total_loss': total_loss,
            'magnitude_loss': magnitude_loss,
            'phase_loss': phase_loss,
            'spectral_loss': spectral_loss,
            'time_loss': time_loss,
            'latent_loss': latent_loss
        }
    
    def train_step(self, batch):
        """Single training step optimized for V100×4"""
        try:
            # Unpack batch
            stereo_audio = batch['audio']  # [B, 2, T]
            
            # Convert to complex spectrogram
            complex_specs = []
            magnitude_specs = []
            
            for i in range(stereo_audio.shape[1]):
                complex_spec = to_complex_spec(stereo_audio[:, i])
                magnitude, _ = to_magnitude_phase(complex_spec)
                
                complex_specs.append(complex_spec)
                magnitude_specs.append(magnitude)
            
            # Stack stereo channels
            complex_input = torch.stack(complex_specs, dim=1)
            magnitude_input = torch.stack(magnitude_specs, dim=1).mean(dim=1)
            
            # Separate real and imaginary parts
            real_part = complex_input.real
            imag_part = complex_input.imag
            target_complex_input = torch.stack([real_part, imag_part], dim=2)
            
            # Forward pass (Accelerate handles mixed precision automatically)
            pred_real, pred_imag, pred_latent = self.model(target_complex_input, magnitude_input)
            
            # Compute losses
            losses = self.compute_loss(
                pred_real, pred_imag,
                real_part, imag_part,
                stereo_audio, pred_latent
            )
            
            loss = losses['total_loss']
            
            return losses, loss
            
        except Exception as e:
            if self.is_main_process:
                self.logger.error(f"Error in training step: {e}")
                import traceback
                traceback.print_exc()
            # Return dummy losses to prevent crash
            dummy_loss = torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)
            dummy_losses = {
                'total_loss': dummy_loss,
                'magnitude_loss': dummy_loss.clone(),
                'phase_loss': dummy_loss.clone(),
                'spectral_loss': dummy_loss.clone(),
                'time_loss': dummy_loss.clone(),
                'latent_loss': dummy_loss.clone()
            }
            return dummy_losses, dummy_loss
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch with V100×4 optimization"""
        self.model.train()
        total_losses = {}
        num_batches = 0
        start_time = time.time()
        
        # Update total_steps if not set and this is first epoch
        if epoch == 0 and self.total_steps is None:
            steps_per_epoch = len(dataloader) // self.accumulate_grad_batches
            total_training_steps = steps_per_epoch * 1000
            self.update_total_steps(total_training_steps)
        
        # Use the already-prepared dataloader
        actual_dataloader = dataloader
        
        # Create progress bar only for main process
        if self.is_main_process:
            batch_pbar = tqdm(
                actual_dataloader,
                desc=f"V100×4 Epoch {epoch}",
                leave=False,
                unit="batch",
                dynamic_ncols=True,
                ascii=True
            )
        else:
            batch_pbar = actual_dataloader
        
        for batch_idx, batch in enumerate(batch_pbar):
            if self.is_main_process and batch_idx == 0:
                print(f"🔍 Processing first batch...")
            
            try:
                # Use Accelerate's gradient accumulation context
                with self.accelerator.accumulate(self.model):
                    # Training step
                    losses, loss = self.train_step(batch)
                    
                    # Skip if dummy loss (error occurred)
                    if loss.item() == 0.0 and all(v.item() == 0.0 for v in losses.values()):
                        continue
                    
                    if self.is_main_process and batch_idx == 0:
                        print(f"🔍 Starting backward pass...")
                    
                    # Use Accelerate's backward for V100×4 distributed training
                    self.accelerator.backward(loss)
                    
                    if self.is_main_process and batch_idx == 0:
                        print(f"🔍 Backward completed, sync_gradients: {self.accelerator.sync_gradients}")
                    
                    # Gradient clipping with sync
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    if self.is_main_process and batch_idx == 0:
                        print(f"🔍 Starting optimizer step...")
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    if self.is_main_process and batch_idx == 0:
                        print(f"🔍 Optimization completed!")
                
                # Accumulate losses
                for key, value in losses.items():
                    if key not in total_losses:
                        total_losses[key] = 0.0
                    total_losses[key] += value.item()
                
                num_batches += 1
                
                if self.is_main_process and batch_idx == 0:
                    print(f"🔍 First batch fully completed!")
                
                # Update progress bar with V100 memory info
                if self.is_main_process:
                    current_lr = float(self.scheduler.get_last_lr()[0]) if self.scheduler else float(self.learning_rate)
                    
                    # Get memory usage for GPU 0
                    gpu_mem = "N/A"
                    if torch.cuda.is_available():
                        try:
                            gpu_mem = f"{torch.cuda.memory_allocated(0) / 1024**3:.1f}GB"
                        except:
                            gpu_mem = "N/A"
                    
                    batch_pbar.set_postfix({
                        'loss': f"{losses['total_loss'].item():.4f}",
                        'mag': f"{losses['magnitude_loss'].item():.3f}",
                        'lr': f"{current_lr:.2e}",
                        'gpu': gpu_mem,
                        'gpus': f"{self.accelerator.num_processes}"
                    })
                
                # Periodic logging and memory monitoring
                if self.is_main_process and batch_idx % 100 == 0:
                    current_lr = float(self.scheduler.get_last_lr()[0]) if self.scheduler else float(self.learning_rate)
                    elapsed = time.time() - start_time
                    
                    self.logger.info(
                        f"V100×4 Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                        f"Loss: {losses['total_loss'].item():.4f}, "
                        f"LR: {current_lr:.2e}, "
                        f"Time: {elapsed:.1f}s, "
                        f"GPUs: {self.accelerator.num_processes}"
                    )
                
                # V100 memory management - clear cache more frequently for 16GB cards
                if torch.cuda.is_available() and batch_idx % 50 == 0:
                    self._monitor_gpu_memory()
                    if batch_idx % 200 == 0:  # Major cleanup every 200 batches
                        torch.cuda.empty_cache()
            
            except Exception as e:
                if self.is_main_process:
                    self.logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Close progress bar
        if self.is_main_process:
            batch_pbar.close()
        
        # Average losses
        if num_batches > 0:
            avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        else:
            avg_losses = {
                'total_loss': 0.0,
                'magnitude_loss': 0.0,
                'phase_loss': 0.0,
                'spectral_loss': 0.0,
                'time_loss': 0.0,
                'latent_loss': 0.0
            }
        
        return avg_losses
    
    def validate(self, val_dataloader):
        """Validation step optimized for V100×4"""
        self.model.eval()
        total_val_losses = {}
        num_val_batches = 0
        
        # Prepare validation dataloader
        if not hasattr(val_dataloader, '_accelerate_prepared'):
            val_dataloader = self.accelerator.prepare(val_dataloader)
            val_dataloader._accelerate_prepared = True
        
        # Create validation progress bar
        if self.is_main_process:
            val_pbar = tqdm(
                val_dataloader,
                desc="V100×4 Validation",
                leave=False,
                unit="batch",
                dynamic_ncols=True,
                ascii=True
            )
        else:
            val_pbar = val_dataloader
        
        with torch.no_grad():
            for batch in val_pbar:
                try:
                    # Forward pass only
                    stereo_audio = batch['audio']
                    
                    # Convert to complex spectrogram
                    complex_specs = []
                    magnitude_specs = []
                    
                    for i in range(stereo_audio.shape[1]):
                        complex_spec = to_complex_spec(stereo_audio[:, i])
                        magnitude, _ = to_magnitude_phase(complex_spec)
                        complex_specs.append(complex_spec)
                        magnitude_specs.append(magnitude)
                    
                    complex_input = torch.stack(complex_specs, dim=1)
                    magnitude_input = torch.stack(magnitude_specs, dim=1).mean(dim=1)
                    
                    real_part = complex_input.real
                    imag_part = complex_input.imag
                    target_complex_input = torch.stack([real_part, imag_part], dim=2)
                    
                    pred_real, pred_imag, pred_latent = self.model(target_complex_input, magnitude_input)
                    losses = self.compute_loss(pred_real, pred_imag, real_part, imag_part, stereo_audio, pred_latent)
                    
                    # Accumulate losses
                    for key, value in losses.items():
                        if key not in total_val_losses:
                            total_val_losses[key] = 0.0
                        total_val_losses[key] += value.item()
                    
                    num_val_batches += 1
                    
                    # Update validation progress
                    if self.is_main_process:
                        val_pbar.set_postfix({
                            'val_loss': f"{losses['total_loss'].item():.4f}",
                            'val_mag': f"{losses['magnitude_loss'].item():.3f}"
                        })
                
                except Exception as e:
                    if self.is_main_process:
                        self.logger.error(f"Error in validation batch: {e}")
                    continue
        
        # Close validation progress bar
        if self.is_main_process:
            val_pbar.close()
        
        # Average validation losses
        if num_val_batches > 0:
            avg_val_losses = {f'val_{key}': value / num_val_batches for key, value in total_val_losses.items()}
        else:
            avg_val_losses = {}
        
        return avg_val_losses
    
    def log_epoch(self, epoch, avg_losses, val_losses, best_loss):
        """Log epoch results for V100×4 training"""
        if not self.is_main_process:
            return
        
        # Console logging
        print(f"🚀 V100×4 Epoch {epoch} completed - Loss: {avg_losses['total_loss']:.6f}")
        if val_losses:
            print(f"   📊 Validation Loss: {val_losses.get('val_total_loss', 'N/A')}")
        print("-" * 60)
        
        # WandB logging with V100×4 specific metrics
        if self.wandb_run is not None:
            log_dict = {
                'epoch': epoch,
                'learning_rate': float(self.scheduler.get_last_lr()[0]) if self.scheduler else float(self.learning_rate),
                **{f'train/{k}': v for k, v in avg_losses.items()},
                **{f'val/{k}': v for k, v in val_losses.items()},
                'best_loss': best_loss,
                'num_gpus': self.accelerator.num_processes
            }
            
            # Add V100 specific GPU memory usage
            if torch.cuda.is_available():
                for gpu_id in range(min(4, torch.cuda.device_count())):  # V100×4
                    try:
                        memory_used = torch.cuda.memory_allocated(gpu_id) / 1024**3
                        memory_cached = torch.cuda.memory_reserved(gpu_id) / 1024**3
                        log_dict[f'v100_{gpu_id}/memory_used_gb'] = memory_used
                        log_dict[f'v100_{gpu_id}/memory_cached_gb'] = memory_cached
                        log_dict[f'v100_{gpu_id}/memory_utilization'] = memory_used / 16.0  # V100 16GB
                    except:
                        pass
            
            try:
                wandb.log(log_dict)
            except Exception as e:
                self.logger.warning(f"Failed to log to wandb: {e}")
    
    def save_checkpoint(self, epoch, losses, save_path):
        """Save training checkpoint optimized for V100×4"""
        if not self.is_main_process:
            return
        
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.accelerator.get_state_dict(self.model),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'losses': losses,
                'total_steps': self.total_steps,
                'accelerator_state': {
                    'num_processes': self.accelerator.num_processes,
                    'mixed_precision': str(self.accelerator.mixed_precision)
                },
                'hardware_info': 'V100x4-16GB'
            }
            
            torch.save(checkpoint, save_path, _use_new_zipfile_serialization=False)
            self.logger.info(f"V100×4 checkpoint saved: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint for V100×4"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Load model state
            self.accelerator.load_state_dict(self.model, checkpoint['model_state_dict'])
            
            # Load optimizer and scheduler
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'total_steps' in checkpoint:
                self.total_steps = checkpoint['total_steps']
                self._create_scheduler()
                if self.accelerator and HAS_ACCELERATE:
                    self.scheduler = self.accelerator.prepare(self.scheduler)
            
            epoch = checkpoint.get('epoch', 0)
            
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if hasattr(self.scheduler, '_last_lr') and len(self.scheduler._last_lr) == 0:
                    self.scheduler._last_lr = [float(self.learning_rate)]
            
            losses = checkpoint.get('losses', {})
            
            # Log hardware info if available
            hw_info = checkpoint.get('hardware_info', 'Unknown')
            if self.is_main_process:
                self.logger.info(f"Checkpoint loaded from {checkpoint_path} (Hardware: {hw_info})")
            
            return epoch, losses
            
        except Exception as e:
            if self.is_main_process:
                self.logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return 0, {}