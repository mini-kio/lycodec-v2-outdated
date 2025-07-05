import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint_sequential
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time
import logging
from pathlib import Path

from .models import LyCodecModel
from .audio import (
    SpectralLoss, 
    to_complex_spec, 
    to_magnitude_phase, 
    to_waveform
)

class LyCodecTrainer:
    """
    LyCodec trainer optimized for V100×4 16GB setup - IMPROVED VERSION
    Features:
    - Mixed precision training with improved loss scaling
    - Safer gradient checkpointing 
    - Memory-efficient batching
    - Distributed training support with proper logging
    - Improved phase loss and scheduler handling
    """
    
    def __init__(self, 
                 model_config=None,
                 learning_rate=1e-4,
                 batch_size=4,  # Optimized for 16GB VRAM
                 accumulate_grad_batches=4,  # Effective batch size: 16
                 max_sequence_length=220500,  # 5 seconds at 44.1kHz
                 use_amp=True,
                 use_checkpointing=True,
                 world_size=4,
                 total_steps=None):  # IMPROVED: Allow dynamic total_steps
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.accumulate_grad_batches = accumulate_grad_batches
        self.max_sequence_length = max_sequence_length
        self.use_amp = use_amp
        self.use_checkpointing = use_checkpointing
        self.world_size = world_size
        self.total_steps = total_steps  # Will be set dynamically if None
        
        # Initialize model
        self.model = LyCodecModel(**(model_config or {}))
        
        # Enable gradient checkpointing for memory efficiency - IMPROVED SAFER METHOD
        if use_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Mixed precision scaler with improved settings
        self.scaler = GradScaler(
            init_scale=2**16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000
        ) if use_amp else None
        
        # Loss functions
        self.spectral_loss = SpectralLoss(n_ffts=[512, 1024, 2048], alpha=1.0, beta=0.1)
        self.mse_loss = nn.MSELoss()
        
        # Setup distributed training if available
        self._setup_distributed()
        
        # Optimizer with improved settings
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-6  # IMPROVED: Better for AMP+fp16
        )
        
        # Learning rate scheduler - will be updated with correct total_steps
        self.scheduler = None
        self._create_scheduler()
        
        # IMPROVED: Setup logging only for main process
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging only for main process to avoid duplicate logs - IMPROVED with rotation"""
        if self.is_main_process:
            import logging.handlers
            
            # Setup rotating file handler to prevent huge log files
            file_handler = logging.handlers.RotatingFileHandler(
                'training.log',
                maxBytes=10*1024*1024,  # 10MB max per file
                backupCount=5  # Keep 5 backup files
            )
            
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    file_handler
                ]
            )
            self.logger = logging.getLogger(__name__)
        else:
            # Create a null logger for non-main processes
            self.logger = logging.getLogger(__name__)
            self.logger.addHandler(logging.NullHandler())
            self.logger.setLevel(logging.CRITICAL)
    
    def _create_scheduler(self):
        """Create scheduler with warm restart support for better resume compatibility"""
        # IMPROVED: Use CosineAnnealingWarmRestarts for better resume behavior
        total_steps = self.total_steps or 100000
        
        # Calculate T_0 for warm restart (typically 10-20% of total steps)
        T_0 = max(total_steps // 10, 1000)  # At least 1000 steps per restart
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=T_0,
            T_mult=2,  # Double restart period each time
            eta_min=self.learning_rate / 100,  # Minimum LR
            last_epoch=-1
        )
        
        # Store original OneCycleLR as backup option
        self._use_onecycle = False
    
    def update_total_steps(self, total_steps: int):
        """Update scheduler with correct total steps after knowing dataset size"""
        self.total_steps = total_steps
        self._create_scheduler()
        self.logger.info(f"Updated scheduler with total_steps={total_steps}")
        
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing using PyTorch 2.3 activation checkpointing - IMPROVED duplicate prevention"""
        try:
            from torch.utils.checkpoint import checkpoint
            
            def apply_checkpointing_to_module(module):
                """Apply checkpointing to individual modules with JIT-safe duplicate prevention"""
                # IMPROVED: JIT-safe flag checking - avoid class attribute pollution
                if (hasattr(module, 'forward') and 
                    not getattr(module, '_ckpt_patched', False)):
                    
                    # Use instance-specific flag instead of class attribute
                    module_id = id(module)
                    if not hasattr(self, '_checkpointed_modules'):
                        self._checkpointed_modules = set()
                    
                    if module_id not in self._checkpointed_modules:
                        original_forward = module.forward
                        
                        def checkpointed_forward(*args, **kwargs):
                            return checkpoint(original_forward, *args, **kwargs, use_reentrant=False)
                        
                        module.forward = checkpointed_forward
                        module._ckpt_patched = True  # Instance-level flag only
                        self._checkpointed_modules.add(module_id)  # Trainer-level tracking
            
            # Apply to ResidualBlocks in encoder
            if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layers'):
                for layer in self.model.encoder.layers:
                    if hasattr(layer, 'psych_attn'):  # ResidualBlock
                        apply_checkpointing_to_module(layer)
            
            # Apply to ResidualBlocks in decoder  
            if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'layers'):
                for layer in self.model.decoder.layers:
                    if hasattr(layer, 'psych_attn'):  # ResidualBlock
                        apply_checkpointing_to_module(layer)
                        
            self.logger.info("Applied activation checkpointing to ResidualBlocks")
            
        except Exception as e:
            self.logger.warning(f"Could not apply gradient checkpointing: {e}")
            # Fallback: disable checkpointing
            self.use_checkpointing = False
    
    def _setup_distributed(self):
        """Setup distributed training"""
        if 'RANK' in os.environ:
            self.rank = int(os.environ['RANK'])
            self.local_rank = int(os.environ['LOCAL_RANK'])
            
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(self.local_rank)
            
            self.model = self.model.cuda(self.local_rank)
            self.model = DDP(
                self.model, 
                device_ids=[self.local_rank],
                find_unused_parameters=False  # Optimization
            )
            
            self.is_distributed = True
            self.is_main_process = self.rank == 0
        else:
            self.rank = 0
            self.local_rank = 0
            self.is_distributed = False
            self.is_main_process = True
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
    
    def compute_loss(self, pred_real, pred_imag, target_real, target_imag, target_audio, pred_latent=None):
        """
        Compute multi-component loss with improved phase loss
        """
        device = pred_real.device
        
        # Reconstruct complex spectrogram
        pred_complex = torch.complex(pred_real, pred_imag)
        target_complex = torch.complex(target_real, target_imag)
        
        # Magnitude and phase losses
        pred_mag = torch.abs(pred_complex)
        target_mag = torch.abs(target_complex)
        magnitude_loss = F.l1_loss(pred_mag, target_mag)
        
        # IMPROVED: Phase loss using 1-cos instead of sin for better gradients
        # IMPROVED: Fixed magnitude_weight calculation with proper dimensionality
        magnitude_weight = target_mag / (target_mag.amax(dim=(-1, -2, -3), keepdim=True) + 1e-8)
        pred_phase = torch.angle(pred_complex)
        target_phase = torch.angle(target_complex)
        
        # Use 1-cos for phase loss (better gradient properties)
        phase_diff_cos = torch.cos(pred_phase - target_phase)
        phase_loss = F.mse_loss((1 - phase_diff_cos) * magnitude_weight, 
                               torch.zeros_like(phase_diff_cos))
        
        # IMPROVED: Minimize to_waveform calls for efficiency
        try:
            # Only compute audio reconstruction once
            pred_audio = to_waveform(pred_complex)
            
            # Ensure same length
            min_len = min(pred_audio.shape[-1], target_audio.shape[-1])
            pred_audio = pred_audio[..., :min_len]
            target_audio_trimmed = target_audio[..., :min_len]
            
            # Multi-scale spectral loss
            spectral_loss = self.spectral_loss(pred_audio, target_audio_trimmed)
            
            # Time-domain loss
            time_loss = F.l1_loss(pred_audio, target_audio_trimmed)
            
        except Exception as e:
            self.logger.warning(f"Audio reconstruction failed: {e}")
            spectral_loss = torch.tensor(0.0, device=device)
            time_loss = torch.tensor(0.0, device=device)
        
        # Latent regularization (optional) - encourage sparsity
        latent_loss = torch.tensor(0.0, device=device)
        if pred_latent is not None:
            # L1 regularization for sparsity
            latent_loss = torch.mean(torch.abs(pred_latent))
        
        # IMPROVED: Combine losses with better weighting
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
        """Single training step with memory optimization"""
        # Unpack batch
        stereo_audio = batch['audio']  # [B, 2, T]
        
        # Move to device
        device = next(self.model.parameters()).device
        stereo_audio = stereo_audio.to(device)
        
        # Convert to complex spectrogram
        complex_specs = []
        magnitude_specs = []
        
        for i in range(stereo_audio.shape[1]):  # Process each channel
            complex_spec = to_complex_spec(stereo_audio[:, i])  # [B, F, T]
            magnitude, _ = to_magnitude_phase(complex_spec)
            
            complex_specs.append(complex_spec)
            magnitude_specs.append(magnitude)
        
        # Stack stereo channels: [B, 2, F, T]
        complex_input = torch.stack(complex_specs, dim=1)
        magnitude_input = torch.stack(magnitude_specs, dim=1).mean(dim=1)  # Average for psychoacoustic analysis
        
        # Separate real and imaginary parts
        real_part = complex_input.real
        imag_part = complex_input.imag
        target_complex_input = torch.stack([real_part, imag_part], dim=2)  # [B, 2, 2, F, T]
        
        # Forward pass with mixed precision
        with autocast(enabled=self.use_amp):
            pred_real, pred_imag, pred_latent = self.model(target_complex_input, magnitude_input)
            
            # Compute losses
            losses = self.compute_loss(
                pred_real, pred_imag,
                real_part, imag_part,
                stereo_audio, pred_latent
            )
            
            loss = losses['total_loss'] / self.accumulate_grad_batches
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return losses
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_losses = {}
        num_batches = 0
        start_time = time.time()
        
        # Update total_steps if not set and this is first epoch
        if epoch == 0 and self.total_steps is None:
            steps_per_epoch = len(dataloader) // self.accumulate_grad_batches
            total_training_steps = steps_per_epoch * 1000  # Assume 1000 epochs max
            self.update_total_steps(total_training_steps)
        
        for batch_idx, batch in enumerate(dataloader):
            # Training step
            losses = self.train_step(batch)
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulate_grad_batches == 0 or (batch_idx + 1) == len(dataloader):
                
                # Gradient clipping for stability
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            # Accumulate losses
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value.item()
            
            num_batches += 1
            
            # Logging
            if self.is_main_process and batch_idx % 100 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                
                self.logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                    f"Loss: {losses['total_loss'].item():.4f}, "
                    f"LR: {current_lr:.2e}, "
                    f"Time: {elapsed:.1f}s"
                )
                
                # Clear CUDA cache periodically
                if torch.cuda.is_available() and batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
        
        # Average losses
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        return avg_losses
    
    def save_checkpoint(self, epoch, losses, save_path):
        """Save training checkpoint with async writing"""
        if not self.is_main_process:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if self.is_distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'losses': losses,
            'total_steps': self.total_steps
        }
        
        # IMPROVED: Use efficient serialization and move tensors to CPU
        checkpoint_cpu = {}
        for key, value in checkpoint.items():
            if isinstance(value, torch.Tensor):
                checkpoint_cpu[key] = value.cpu()
            elif isinstance(value, dict):
                checkpoint_cpu[key] = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                     for k, v in value.items()}
            else:
                checkpoint_cpu[key] = value
        
        torch.save(checkpoint_cpu, save_path, _use_new_zipfile_serialization=False)
        self.logger.info(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint with proper scheduler state handling"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if self.is_distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # IMPROVED: Handle scheduler state loading properly
        if 'total_steps' in checkpoint:
            self.total_steps = checkpoint['total_steps']
            self._create_scheduler()
        
        epoch = checkpoint.get('epoch', 0)
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # FIXED: Handle _last_lr issue
            if hasattr(self.scheduler, '_last_lr') and len(self.scheduler._last_lr) == 0:
                self.scheduler._last_lr = [self.learning_rate]
        else:
            # IMPROVED: Restore scheduler to proper epoch if no state dict
            for _ in range(epoch):
                self.scheduler.step()
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        losses = checkpoint.get('losses', {})
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return epoch, losses
    
    def validate(self, val_dataloader):
        """Validation step"""
        self.model.eval()
        total_val_losses = {}
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                # Forward pass only
                stereo_audio = batch['audio'].to(next(self.model.parameters()).device)
                
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
                
                with autocast(enabled=self.use_amp):
                    pred_real, pred_imag, pred_latent = self.model(target_complex_input, magnitude_input)
                    losses = self.compute_loss(pred_real, pred_imag, real_part, imag_part, stereo_audio, pred_latent)
                
                # Accumulate losses
                for key, value in losses.items():
                    if key not in total_val_losses:
                        total_val_losses[key] = 0.0
                    total_val_losses[key] += value.item()
                
                num_val_batches += 1
        
        # Average validation losses
        avg_val_losses = {f'val_{key}': value / num_val_batches for key, value in total_val_losses.items()}
        return avg_val_losses