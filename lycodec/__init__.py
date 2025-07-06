"""
LyCodec - High-Quality Stereo Audio Codec with Psychoacoustic Features
f10c10 compression (100x) with phase preservation and perceptual optimization

v2.0 Major Features:
- Real-time streaming decoder with ~100ms latency
- V100×4 16GB optimized training with WandB logging
- Enhanced multi-GPU stability and memory management
- Improved scheduler with warm restart support
- JIT-safe gradient checkpointing
- Cached import optimization for better startup time
- Progress tracking with tqdm for training visibility
- Configurable dataset limits and improved distributed logging
"""

from .inference import LyCodec
from .models import LyEncoder, LyDecoder, PsychoacousticTransform, LinearAttention
from .audio import (
    gammatone_filterbank, 
    psychoacoustic_masking, 
    high_quality_resample, 
    to_magnitude_phase, 
    from_magnitude_phase,
    to_complex_spec,
    to_waveform,
    normalize_audio
)
from .training import LyCodecTrainer
from .streaming import StreamingDecoder, create_streaming_decoder

__version__ = "2.0"
__author__ = "Mini_kio"

__all__ = [
    'LyCodec',
    'LyEncoder', 
    'LyDecoder',
    'PsychoacousticTransform',
    'LinearAttention',
    'LyCodecTrainer',
    'StreamingDecoder',
    'create_streaming_decoder',
    'gammatone_filterbank',
    'psychoacoustic_masking',
    'high_quality_resample',
    'to_magnitude_phase',
    'from_magnitude_phase',
    'to_complex_spec',
    'to_waveform',
    'normalize_audio'
]