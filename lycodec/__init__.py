"""
LyCodec v2.5: Production-Ready 44.1kHz Stereo Audio Codec with Continuous Latent Space
===================================================================================

Public API for LyCodec v2.5 audio codec with f10c10 continuous compression.
Implements semantic-preserving compression with smooth latent manifolds.
"""

__version__ = "2.5.0"
__author__ = "Mini kio"

import torch
from .model import LyCodecV25Model, LyCodecV25Config
from .psychoacoustic import PsychoacousticTransform, FastRMSNorm2D
from .vocoder import DDSPVocoder, HarmonicSynthesizer
from .utils import (
    ProductionAdaptiveBitAllocator,
    StableCheckpointer,
    AdaptiveGradientClipper,
    AudioDataProcessor
)

# Production constants for 44.1kHz stereo with f10c10 compression
SAMPLE_RATE = 44100
CHANNELS = 2
SEGMENT_LENGTH = 5.0  # seconds
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_LENGTH)  # 220,500 samples
SEGMENTS_PER_TRACK = 3

# f10c10 compression specifications
COMPRESSION_RATIO = 100  # f10c10 = 100:1 compression
LATENT_CHANNELS = 64
LATENT_LENGTH = SEGMENT_SAMPLES // COMPRESSION_RATIO  # 2205 samples

# Model architecture constants
MODEL_PARAMS = 55_000_000  # ~55M parameters for v2.5
TARGET_BITRATE_RANGE = (64, 256)  # kbps adaptive range (ultra-low bitrate)
HARMONICS_COUNT = 48
NOISE_BANDS = 8

# Training configuration
V100_GPU_COUNT = 4
GPU_MEMORY_GB = 16
INFERENCE_MEMORY_TARGET_GB = 6  # Lower target due to continuous space efficiency

def load_model(checkpoint_path: str = None, device: str = "cuda") -> LyCodecV25Model:
    """
    Load LyCodec v2.5 model with continuous latent space configuration.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Target device for inference
        
    Returns:
        Configured LyCodecV25Model instance
    """
    config = LyCodecV25Config()
    model = LyCodecV25Model(config)
    
    if checkpoint_path:
        model.load_checkpoint(checkpoint_path)
    
    return model.to(device)

def encode(audio: torch.Tensor, model: LyCodecV25Model, bitrate: int = 128) -> torch.Tensor:
    """
    Encode stereo audio to continuous latent representation.
    
    Args:
        audio: Input tensor [batch, channels=2, samples]
        model: Trained LyCodec v2.5 model
        bitrate: Target bitrate in kbps
        
    Returns:
        Continuous latent tensor [batch, latent_channels, latent_length]
    """
    return model.encode(audio, bitrate)

def decode(latent: torch.Tensor, model: LyCodecV25Model) -> torch.Tensor:
    """
    Decode continuous latent representation to stereo audio.
    
    Args:
        latent: Continuous latent tensor [batch, latent_channels, latent_length]
        model: Trained LyCodec v2.5 model
        
    Returns:
        Reconstructed audio tensor [batch, channels=2, samples]
    """
    return model.decode(latent)

def interpolate_latents(latent1: torch.Tensor, latent2: torch.Tensor, 
                       alpha: float, model: LyCodecV25Model) -> torch.Tensor:
    """
    Semantically-aware interpolation in continuous latent space.
    
    Args:
        latent1: First latent [batch, latent_channels, latent_length]
        latent2: Second latent [batch, latent_channels, latent_length]
        alpha: Interpolation factor [0, 1]
        model: Trained LyCodec v2.5 model
        
    Returns:
        Interpolated latent with semantic consistency
    """
    return model.semantic_interpolation(latent1, latent2, alpha)

__all__ = [
    # Core model
    "LyCodecV25Model", "LyCodecV25Config",
    # Components
    "PsychoacousticTransform", "FastRMSNorm2D",
    "DDSPVocoder", "HarmonicSynthesizer",
    # Utilities
    "ProductionAdaptiveBitAllocator", "StableCheckpointer",
    "AdaptiveGradientClipper", "AudioDataProcessor",
    # API functions
    "load_model", "encode", "decode", "interpolate_latents",
    # Constants
    "SAMPLE_RATE", "CHANNELS", "SEGMENT_LENGTH", "SEGMENT_SAMPLES",
    "SEGMENTS_PER_TRACK", "COMPRESSION_RATIO", "LATENT_CHANNELS", "LATENT_LENGTH",
    "MODEL_PARAMS", "TARGET_BITRATE_RANGE"
]