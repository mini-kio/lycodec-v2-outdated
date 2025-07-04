"""
LyCodec v2.1: Production-Ready 44.1kHz Stereo Audio Codec
========================================================

Public API for LyCodec audio codec with production-grade architecture.
Optimized for V100×4 16GB training and <8GB inference deployment.
"""

__version__ = "2.1.0"
__author__ = "LyCodec Team"

import torch
from .model import LyCodecModel, LyCodecConfig
from .quantization import VectorizedQuantizer, ConsistencyAwareNoiseScheduler
from .psychoacoustic import PsychoacousticTransform, FastRMSNorm2D
from .vocoder import DDSPVocoder, HarmonicSynthesizer
from .utils import (
    ProductionAdaptiveBitAllocator,
    StableCheckpointer,
    AdaptiveGradientClipper,
    AudioDataProcessor
)

# Production constants for 44.1kHz stereo
SAMPLE_RATE = 44100
CHANNELS = 2
SEGMENT_LENGTH = 5.0  # seconds
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_LENGTH)  # 220,500 samples
SEGMENTS_PER_TRACK = 3

# Model architecture constants
MODEL_PARAMS = 45_000_000  # ~45M parameters
TARGET_BITRATE_RANGE = (128, 320)  # kbps adaptive range
HARMONICS_COUNT = 48
NOISE_BANDS = 8

# Training configuration
V100_GPU_COUNT = 4
GPU_MEMORY_GB = 16
INFERENCE_MEMORY_TARGET_GB = 8

def load_model(checkpoint_path: str = None, device: str = "cuda") -> LyCodecModel:
    """
    Load LyCodec model with production-ready configuration.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Target device for inference
        
    Returns:
        Configured LyCodecModel instance
    """
    config = LyCodecConfig()
    model = LyCodecModel(config)
    
    if checkpoint_path:
        model.load_checkpoint(checkpoint_path)
    
    return model.to(device)

def encode(audio: torch.Tensor, model: LyCodecModel, bitrate: int = 192) -> bytes:
    """
    Encode stereo audio to compressed bitstream.
    
    Args:
        audio: Input tensor [batch, channels=2, samples]
        model: Trained LyCodec model
        bitrate: Target bitrate in kbps
        
    Returns:
        Compressed audio bitstream
    """
    return model.encode(audio, bitrate)

def decode(bitstream: bytes, model: LyCodecModel) -> torch.Tensor:
    """
    Decode compressed bitstream to stereo audio.
    
    Args:
        bitstream: Compressed audio data
        model: Trained LyCodec model
        
    Returns:
        Reconstructed audio tensor [batch, channels=2, samples]
    """
    return model.decode(bitstream)

__all__ = [
    # Core model
    "LyCodecModel", "LyCodecConfig",
    # Components
    "VectorizedQuantizer", "ConsistencyAwareNoiseScheduler",
    "PsychoacousticTransform", "FastRMSNorm2D",
    "DDSPVocoder", "HarmonicSynthesizer",
    # Utilities
    "ProductionAdaptiveBitAllocator", "StableCheckpointer",
    "AdaptiveGradientClipper", "AudioDataProcessor",
    # API functions
    "load_model", "encode", "decode",
    # Constants
    "SAMPLE_RATE", "CHANNELS", "SEGMENT_LENGTH", "SEGMENT_SAMPLES",
    "SEGMENTS_PER_TRACK", "MODEL_PARAMS", "TARGET_BITRATE_RANGE"
]