# LyCodec v2.0 - High-Quality Stereo Audio Codec

🎵 **f10c10 compression (100x)** with phase preservation and perceptual optimization

## ✨ Key Features

- **Real-time streaming decoder** with ~100ms latency
- **V100×4 16GB optimized training** with WandB logging  
- **Enhanced multi-GPU stability** and memory management
- **Improved scheduler** with warm restart support
- **JIT-safe gradient checkpointing** for memory efficiency
- **Cached import optimization** for better startup time
- **Robust DSP implementation** with streaming-optimized STFT/ISTFT

## 🚀 Recent Improvements (v2.0)

### DSP & Signal Processing
- ✅ **STFT/ISTFT consistency**: Fixed `center=False` and `pad_mode='reflect'` for streaming
- ✅ **High-quality resampling**: pysoxr → torchaudio → scipy fallback with caching
- ✅ **Dynamic range compression**: Properly documented as inference-only with `torch.no_grad()`
- ✅ **Phase preservation**: Improved magnitude/phase decomposition (now in public API)

### PyTorch & Training  
- ✅ **Accelerate Integration**: Simplified distributed training with Hugging Face Accelerate
- ✅ **Triton GPU Kernels**: Custom optimized kernels for 2-3x speedup on CUDA
- ✅ **Gradient checkpointing**: Instance-level patching to avoid JIT/class attribute issues
- ✅ **FP16 safety**: Enhanced CPU guards with clear warnings for numpy compatibility
- ✅ **Scheduler improvements**: CosineAnnealingWarmRestarts for better resume compatibility
- ✅ **Decoder activation**: Added tanh activation to prevent phase wrap-around
- ✅ **Progress tracking**: Added tqdm progress bars for training visibility
- ✅ **Dataset optimization**: Configurable file limits and improved distributed logging

### Streaming & Real-time
- ✅ **Streaming decoder**: Overlap-add with fade windows and real-time queue handling
- ✅ **Parameter validation**: Chunk/overlap length mismatch guards and IndexError prevention
- ✅ **Memory leak monitoring**: GPU memory tracking in inference paths

### Multi-GPU & Distributed
- ✅ **Accelerate Support**: Easy distributed training without complex DDP setup
- ✅ **WandB integration**: Experiment tracking with GPU memory and training metrics
- ✅ **V100×4 optimization**: Hardware-specific settings and memory management

## 📦 Installation

```

### Quick Setup for Multi-GPU

```bash
# 1. Install dependencies with Accelerate
pip install -r requirements.txt

# 2. Configure Accelerate (interactive setup)
accelerate config

# 3. Start training
accelerate launch train.py --config config.yaml
```

**Accelerate Benefits:**
- ✅ **No complex environment variables** - automated setup
- ✅ **Automatic mixed precision** - FP16/BF16 handled seamlessly  
- ✅ **Better error handling** - graceful fallbacks
- ✅ **Cross-platform compatibility** - works on various GPU setupsbash
# Install dependencies
pip install -r requirements.txt

# For high-quality resampling (recommended)
pip install soxr

# For distributed training
pip install wandb
```

## 🎯 Quick Start

### Basic Encoding/Decoding

```python
from lycodec import LyCodec

# Initialize codec
codec = LyCodec(device='cuda', half_precision=True)

# Encode audio file
latent = codec.encode('input.wav', normalize=True)
print(f"Compression: {latent.shape}")  # f10c10 compressed

# Decode back to audio
audio = codec.decode(latent)
codec.save_audio(audio, 'output.wav')
```

### Streaming Decoder

```python
from lycodec import create_streaming_decoder

# Create streaming decoder with 100ms latency
decoder = create_streaming_decoder('model.pth', latency_ms=100)

# Stream decode with context manager
with decoder.streaming_session():
    for latent_chunk in latent_stream:
        if decoder.put_latent_chunk(latent_chunk):
            audio_chunk = decoder.get_audio_chunk()
            if audio_chunk:
                # Process audio chunk...
                play_audio(audio_chunk['audio'])
```

### Advanced API Usage

```python
from lycodec import to_magnitude_phase, from_magnitude_phase, high_quality_resample

# Magnitude/phase processing
complex_spec = stft_transform(audio)
magnitude, phase = to_magnitude_phase(complex_spec)

# High-quality resampling (automatic backend selection)
resampled = high_quality_resample(audio, 44100, 22050)
```

## 🏋️‍♀️ Training

### Single GPU

```python
from lycodec import LyCodecTrainer

trainer = LyCodecTrainer(
    batch_size=4,
    use_amp=True,
    use_checkpointing=True
)

trainer.train(dataloader, epochs=100)
```

### Multi-GPU (V100×4) with Accelerate

```bash
# Install Accelerate
pip install accelerate

# Configure Accelerate (one-time setup)
accelerate config

# Launch distributed training with Accelerate
accelerate launch train.py --config config.yaml
```

#### Accelerate Configuration Example
```
In which compute environment are you running? This machine
Which type of machine are you using? Multi-GPU
How many different machines will you use? 1
Do you want to use DeepSpeed? No
Do you want to use FullyShardedDataParallel? No
How many GPU(s) should be used for distributed training? 4
Do you wish to use FP16 or BF16 (mixed precision)? fp16
```

### Configuration (config.yaml)

```yaml
# Hardware optimized for V100×4 16GB with Accelerate
model:
  latent_dim: 64
  base_channels: 64
  n_layers: 6

training:
  batch_size: 4              # Per GPU
  accumulate_grad_batches: 4 # Effective: 64 total
  learning_rate: 1e-4
  max_epochs: 100
  mixed_precision: true      # Handled by Accelerate

# WandB experiment tracking
wandb:
  project: "lycodec-v100x4"
  entity: "your-team"
  
# Data settings
data:
  data_dir: "dataset/raw/music"
  file_limit: null              # null = use all audio files
  segment_seconds: 5.0
  samples_per_track: 3

# Accelerate handles GPU configuration automatically
accelerate:
  mixed_precision: "fp16"
  gradient_accumulation_steps: 4
  num_processes: 4
```

## 🔬 Testing & Validation

### API Improvements Test

```bash
python test_api.py
```

Tests new safety guards, FP16 CPU handling, and streaming parameter validation.

### Round-trip Quality Test

```bash
python test_codec.py --model model.pth --audio test.wav
```

Measures SNR, compression ratio, and reconstruction quality.

### Compression Analysis

```bash
python analyze_compression.py
```

Analyzes true compression ratios including file I/O overhead and metadata.

### Streaming Demo

```bash
python demo_streaming.py --model model.pth --audio test.wav --latency 100
```

Demonstrates real-time streaming decode with configurable latency.

## 🔧 Advanced Configuration

### Memory Optimization

```python
# For <6GB GPUs
codec = LyCodec(
    device='cuda',
    half_precision=True,
    max_chunk_length=110250,  # 2.5s chunks
    overlap_length=2205       # 0.05s overlap
)

# CPU fallback with warnings
codec = LyCodec(device='cpu', half_precision=False)
```

### Accelerate Configuration

```bash
# Configure for different setups
accelerate config

# Single GPU
# Multi-GPU (same machine)
# Multi-node (multiple machines)
# CPU-only training
```

### Streaming Parameters

```python
# Ultra-low latency (requires more CPU)
decoder = StreamingDecoder(
    model_path='model.pth',
    chunk_size=2205,     # 50ms chunks
    overlap_ratio=0.25,  # 25% overlap
    buffer_size=4        # Small buffer
)

# High quality (higher latency)
decoder = StreamingDecoder(
    model_path='model.pth', 
    chunk_size=8820,     # 200ms chunks
    overlap_ratio=0.5,   # 50% overlap
    buffer_size=16       # Large buffer
)
```

## 🛠️ Development

### Safety Guards

The codec includes extensive safety guards:

- **Parameter validation**: Chunk sizes, overlap ratios, latency bounds
- **Memory monitoring**: GPU memory leak detection and peak tracking  
- **FP16 safety**: CPU compatibility warnings and automatic fallbacks
- **Gradient safety**: Inference-only functions use `torch.no_grad()`
- **JIT compatibility**: Instance-level checkpointing flags
- **Accelerate integration**: Simplified distributed training setup

### Known Limitations

- **Streaming latency**: Current demo is offline-encode + streaming-decode
- **Compression ratio**: Includes storage overhead; real deployment may use bitpacking
- **Phase wrapping**: Bounded by tanh activation; may affect very loud signals
- **CPU FP16**: Disabled for stability; may affect numpy interop

### Contributing

1. Run tests: `python test_api.py && python test_codec.py`
2. Check safety: All new streaming parameters must validate
3. Memory: Use context managers for GPU memory management
4. Docs: Update docstrings for inference-only functions
5. **Accelerate**: Use `accelerate launch` for distributed training

## 📄 License

Licensed under the Apache License 2.0. See LICENSE file for details.

---

**LyCodec v2.0** - Ready for production training and streaming deployment! 🎉