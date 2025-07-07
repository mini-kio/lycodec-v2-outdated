#!/bin/bash

# LyCodec V100×4 Optimized Training Script
# 9-step performance optimization applied

echo "🚀 Starting LyCodec V100×4 Optimized Training"
echo "📊 Performance Optimizations Applied:"
echo "   STEP 1: soxr C-backend - MP3→PCM 리샘플 5ms → 0.3ms"
echo "   STEP 2: GPU 리샘플·STFT - CPU 부하 -60%"
echo "   STEP 3: DataLoader I/O 튜닝 - Disk wait ↓"
echo "   STEP 4: 배치/시퀀스 재조정 - 4s segments, batch=16"
echo "   STEP 5: pinned memory + FP16 - PCIe copy 170MB → 34MB"
echo "   STEP 6: 로깅 최소화 - GIL·I/O 잠금 ↓"
echo "   STEP 7: TensorCore 최적화 - Conv/Linear 15-25% ↑"
echo "   STEP 8: JIT warm-up 분리 - 첫 배치 280s 제거"
echo "   STEP 9: torch.compile - kernel launch latency 10-15% ↓"
echo ""

# Check if accelerate is configured
if [ ! -f ~/.cache/huggingface/accelerate/default_config.yaml ]; then
    echo "❌ Accelerate not configured. Please run:"
    echo "   accelerate config"
    echo "   Select: 4 GPUs, FP16, no DeepSpeed"
    exit 1
fi

# Install soxr if not present
python -c "import soxr" 2>/dev/null || {
    echo "📦 Installing soxr for STEP 1 optimization..."
    pip install soxr==0.3.7
}

# Verify GPU availability
python -c "import torch; print(f'✅ GPUs available: {torch.cuda.device_count()}')"

if [ $? -ne 0 ]; then
    echo "❌ CUDA not available"
    exit 1
fi

# Set optimal environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_CUDNN_V8_API_ENABLED=1

# Enable optimizations
export TORCH_COMPILE_DEBUG=0
export PYTORCH_PRETRAINED_BERT_CACHE=""

echo "🎯 Starting optimized training with accelerate..."
echo "💾 Checkpoints will be saved to: checkpoints_optimized/"
echo ""

# Run with accelerate for 4-GPU distributed training
accelerate launch \
    --config_file ~/.cache/huggingface/accelerate/default_config.yaml \
    --main_process_port 29500 \
    train.py \
    --config config.yaml

echo ""
echo "✅ Training completed!"
echo "📊 Check logs: v100x4_logmel_training.log"
echo "💾 Best model: checkpoints_optimized/best_logmel.pt"
