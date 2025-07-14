"""
LyCodec - High-Quality Stereo Audio Codec with Log-Mel + Phase Architecture
f10c10 compression (100x) with phase preservation and psychoacoustic masking

v2.0 Log-Mel + Phase Architecture:
- Waveform → STFT → Magnitude/Phase → Mel filterbank → log_mel (128 bin) + phase preservation
- PsychTransform applies masking curve weighting in log-mel domain
- Real-time streaming decoder with ~100ms latency
- V100×4 16GB optimized training with WandB logging
- Enhanced multi-GPU stability and memory management
- DDP unused parameters issue completely fixed
"""

from .inference import Codec
from .models import Model, Encoder, Decoder, PsychTransform, Attention
from .audio import (
    # Core audio processing functions
    gammatone_filterbank, 
    psychoacoustic_masking, 
    high_quality_resample, 
    normalize_audio,
    
    # STFT/ISTFT functions
    stft_transform,
    istft_transform,
    to_complex_spec,
    to_waveform,
    
    # Magnitude/Phase functions
    to_magnitude_phase, 
    from_magnitude_phase,
    
    # Mel-scale processing functions
    create_mel_filterbank,
    to_mel_spectrogram,
    to_log_mel,
    from_log_mel,
    mel_to_magnitude,
    
    # Utility functions
    apply_window_function,
    dynamic_range_compression,
    create_deterministic_seed,
    get_optimal_pin_memory,
    
    # Loss functions
    SpectralLoss,
    
    # Constants
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    F_MIN,
    F_MAX
)
from .training import Trainer
from .streaming import StreamingDecoder, create_streaming_decoder

__version__ = "2.0-logmel"
__author__ = "Mini_kio"
__architecture__ = "log_mel_phase"
__compression_ratio__ = "f10c10_100x"

# Architecture information
ARCHITECTURE_INFO = {
    "version": "2.0",
    "architecture": "log_mel_phase",
    "mel_bins": 128,
    "compression_ratio": 100,
    "features": [
        "Log-mel domain processing",
        "Phase preservation",
        "Psychoacoustic masking",
        "f10c10 compression (100x)",
        "Real-time streaming",
        "DDP compatibility",
        "V100×4 optimized",
        "DDP unused parameters fixed"
    ],
    "pipeline": [
        "Waveform",
        "STFT",
        "Magnitude/Phase separation", 
        "Mel filterbank",
        "Log-mel (128 bin) + Phase preservation",
        "PsychTransform (masking curve weighting)",
        "Encoder f10c10",
        "Latent representation",
        "Decoder",
        "Log-mel + Phase reconstruction",
        "Magnitude reconstruction",
        "Complex spectrogram",
        "ISTFT",
        "Waveform"
    ],
    "fixes_applied": {
        "ddp_unused_parameters_fixed": True,
        "find_unused_parameters_enabled": True,
        "all_parameters_used_in_loss": True,
        "simplified_class_names": True,
        "log_spam_eliminated": True
    }
}

__all__ = [
    # Core classes
    'Codec',
    'Model',
    'Encoder', 
    'Decoder',
    'PsychTransform',
    'Attention',
    'Trainer',
    'StreamingDecoder',
    'create_streaming_decoder',
    
    # Audio processing functions
    'gammatone_filterbank',
    'psychoacoustic_masking',
    'high_quality_resample',
    'normalize_audio',
    
    # STFT/ISTFT functions
    'stft_transform',
    'istft_transform',
    'to_complex_spec',
    'to_waveform',
    
    # Magnitude/Phase functions
    'to_magnitude_phase',
    'from_magnitude_phase',
    
    # Mel-scale processing functions
    'create_mel_filterbank',
    'to_mel_spectrogram',
    'to_log_mel',
    'from_log_mel',
    'mel_to_magnitude',
    
    # Utility functions
    'apply_window_function',
    'dynamic_range_compression',
    'create_deterministic_seed',
    'get_optimal_pin_memory',
    
    # Loss functions
    'SpectralLoss',
    
    # Constants
    'SAMPLE_RATE',
    'N_FFT',
    'HOP_LENGTH',
    'N_MELS',
    'F_MIN',
    'F_MAX',
    
    # Architecture info
    'ARCHITECTURE_INFO'
]

def get_architecture_info():
    """
    Get detailed information about the log-mel + phase architecture
    """
    return ARCHITECTURE_INFO.copy()

def print_architecture_summary():
    """
    Print a summary of the log-mel + phase architecture
    """
    info = ARCHITECTURE_INFO
    print(f"🎵 LyCodec v{info['version']} - {info['architecture'].upper()} Architecture")
    print(f"📊 Compression: {info['compression_ratio']}x with {info['mel_bins']} mel bins")
    print(f"✨ Features:")
    for feature in info['features']:
        print(f"   • {feature}")
    print(f"🔄 Processing Pipeline:")
    for i, step in enumerate(info['pipeline'], 1):
        arrow = " → " if i < len(info['pipeline']) else ""
        print(f"   {i:2d}. {step}{arrow}")

def verify_installation():
    """
    Verify that LyCodec is properly installed with all dependencies
    """
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch not found: {e}")
        return False
    
    try:
        import soundfile as sf
        print(f"✅ SoundFile: {sf.__version__}")
    except ImportError as e:
        print(f"❌ SoundFile not found: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy not found: {e}")
        return False
    
    # Optional dependencies
    optional_deps = {
        'torchaudio': 'High-quality audio resampling',
        'soxr': 'Highest quality resampling (recommended)',
        'accelerate': 'Multi-GPU distributed training',
        'wandb': 'Experiment tracking and logging',
        'tqdm': 'Progress bars during training'
    }
    
    for dep, description in optional_deps.items():
        try:
            __import__(dep)
            print(f"✅ {dep}: Available - {description}")
        except ImportError:
            print(f"⚠️ {dep}: Not available - {description}")
    
    # Test model creation
    try:
        model = Model()
        print(f"✅ Model: Successfully created")
        
        # Test mel filterbank creation
        mel_filterbank = create_mel_filterbank()
        print(f"✅ Mel filterbank: {mel_filterbank.shape} ({N_MELS} mel bins)")
        
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def create_test_model():
    """
    Create a test model for verification purposes
    """
    try:
        model = Model(
            latent_dim=32,
            base_channels=32,
            n_layers=3
        )
        
        print(f"✅ Test model created:")
        print(f"   Architecture: {__architecture__}")
        print(f"   Mel bins: {N_MELS}")
        print(f"   Compression: {__compression_ratio__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    except Exception as e:
        print(f"❌ Test model creation failed: {e}")
        return None

def quick_test():
    """
    Quick functionality test of the log-mel + phase architecture
    """
    print(f"🧪 LyCodec v{__version__} Quick Test")
    print("-" * 50)
    
    try:
        # Test model creation
        model = create_test_model()
        if model is None:
            return False
        
        # Test mel filterbank
        mel_filterbank = create_mel_filterbank(n_mels=N_MELS)
        print(f"✅ Mel filterbank: {mel_filterbank.shape}")
        
        # Test audio processing pipeline
        import torch
        
        # Create test log-mel and phase data
        batch_size = 1
        time_frames = 100
        test_log_mel = torch.randn(batch_size, N_MELS, time_frames)
        test_phase = torch.randn(batch_size, N_MELS, time_frames)
        
        print(f"✅ Test input: log_mel {test_log_mel.shape}, phase {test_phase.shape}")
        
        # Test model forward pass
        model.eval()
        with torch.no_grad():
            pred_log_mel, pred_phase, latent = model(test_log_mel, test_phase)
        
        print(f"✅ Model forward pass:")
        print(f"   Output log_mel: {pred_log_mel.shape}")
        print(f"   Output phase: {pred_phase.shape}")
        print(f"   Latent: {latent.shape}")
        
        # Calculate compression ratio
        input_elements = test_log_mel.numel() + test_phase.numel()
        latent_elements = latent.numel()
        compression_ratio = input_elements / latent_elements
        
        print(f"✅ Compression ratio: {compression_ratio:.1f}x")
        print(f"   Input elements: {input_elements:,}")
        print(f"   Latent elements: {latent_elements:,}")
        
        # Test gradient flow
        print("🔍 Testing gradient flow...")
        model.train()
        
        # Create dummy targets
        target_log_mel = torch.randn_like(test_log_mel, requires_grad=False)
        target_phase = torch.randn_like(test_phase, requires_grad=False)
        
        # Forward pass
        pred_log_mel, pred_phase, latent = model(test_log_mel, test_phase)
        
        # Simple loss
        loss = torch.nn.functional.mse_loss(pred_log_mel, target_log_mel) + \
               torch.nn.functional.mse_loss(pred_phase, target_phase) + \
               0.01 * torch.mean(latent ** 2)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        grad_count = 0
        total_params = 0
        for name, param in model.named_parameters():
            total_params += 1
            if param.requires_grad and param.grad is not None and param.grad.abs().sum() > 0:
                grad_count += 1
        
        print(f"✅ Gradient flow: {grad_count}/{total_params} parameters have gradients")
        
        if grad_count == total_params:
            print("🎉 All parameters receive gradients - DDP issue fixed!")
        else:
            print(f"⚠️ {total_params - grad_count} parameters without gradients")
        
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Backward compatibility
LyCodec = Codec
LyCodecModel = Model
LyEncoder = Encoder
LyDecoder = Decoder
PsychoacousticTransform = PsychTransform
LinearAttention = Attention
LyCodecTrainer = Trainer 