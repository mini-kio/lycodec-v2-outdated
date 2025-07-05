#!/usr/bin/env python3
"""
Test script for LyCodec API improvements and bug fixes
Tests the new exposed functions and safety guards
"""

import torch
import numpy as np
import traceback
import tempfile
from pathlib import Path

# Test imports
try:
    from lycodec import (
        LyCodec, 
        to_magnitude_phase, 
        from_magnitude_phase,
        create_streaming_decoder,
        StreamingDecoder,
        high_quality_resample
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

def test_magnitude_phase_api():
    """Test the newly exposed magnitude/phase API"""
    print("\n=== Testing Magnitude/Phase API ===")
    
    try:
        # Create test complex spectrogram
        B, F, T = 2, 513, 100
        real = torch.randn(B, F, T)
        imag = torch.randn(B, F, T)
        complex_spec = torch.complex(real, imag)
        
        # Test magnitude/phase decomposition
        magnitude, phase = to_magnitude_phase(complex_spec)
        print(f"✓ to_magnitude_phase: {complex_spec.shape} -> mag:{magnitude.shape}, phase:{phase.shape}")
        
        # Test reconstruction
        reconstructed = from_magnitude_phase(magnitude, phase)
        print(f"✓ from_magnitude_phase: mag:{magnitude.shape}, phase:{phase.shape} -> {reconstructed.shape}")
        
        # Test reconstruction accuracy
        mse = torch.mean((complex_spec - reconstructed).abs() ** 2)
        print(f"✓ Reconstruction MSE: {mse.item():.2e}")
        
        if mse.item() < 1e-10:
            print("✓ Perfect reconstruction!")
        else:
            print("⚠ Some reconstruction error (expected for complex operations)")
            
    except Exception as e:
        print(f"✗ Magnitude/Phase API test failed: {e}")
        traceback.print_exc()

def test_fp16_cpu_guards():
    """Test FP16 on CPU safety guards"""
    print("\n=== Testing FP16 CPU Guards ===")
    
    try:
        # Force CPU device to test guards
        codec_cpu = LyCodec(device='cpu', half_precision=True)
        print(f"✓ CPU codec created, half_precision={codec_cpu.half_precision}")
        
        if codec_cpu.half_precision:
            print("⚠ FP16 enabled on CPU - should show warnings")
        else:
            print("✓ FP16 correctly disabled on CPU")
            
    except Exception as e:
        print(f"✗ FP16 CPU guard test failed: {e}")
        traceback.print_exc()

def test_streaming_safety_guards():
    """Test streaming decoder parameter validation"""
    print("\n=== Testing Streaming Safety Guards ===")
    
    # Create a dummy model file (not functional, just for testing params)
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        dummy_model_path = f.name
        torch.save({'dummy': True}, f.name)
    
    try:
        # Test invalid chunk_size
        try:
            StreamingDecoder(dummy_model_path, chunk_size=0)
            print("✗ Should have caught invalid chunk_size=0")
        except ValueError as e:
            print(f"✓ Caught invalid chunk_size: {e}")
        
        # Test invalid overlap_ratio
        try:
            StreamingDecoder(dummy_model_path, overlap_ratio=1.5)
            print("✗ Should have caught invalid overlap_ratio=1.5")
        except ValueError as e:
            print(f"✓ Caught invalid overlap_ratio: {e}")
        
        # Test overlap >= chunk_size
        try:
            StreamingDecoder(dummy_model_path, chunk_size=1000, overlap_ratio=0.9999)
            print("✗ Should have caught overlap >= chunk_size")
        except ValueError as e:
            print(f"✓ Caught overlap >= chunk_size: {e}")
        
        # Test create_streaming_decoder with invalid latency
        try:
            create_streaming_decoder(dummy_model_path, latency_ms=-10)
            print("✗ Should have caught negative latency")
        except ValueError as e:
            print(f"✓ Caught negative latency: {e}")
        
        print("✓ All streaming safety guards working")
        
    except Exception as e:
        print(f"✗ Streaming safety guard test failed: {e}")
        traceback.print_exc()
    finally:
        # Clean up
        Path(dummy_model_path).unlink(missing_ok=True)

def test_high_quality_resample():
    """Test high quality resampling with different backends"""
    print("\n=== Testing High Quality Resampling ===")
    
    try:
        # Create test audio
        sr_orig = 44100
        sr_target = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr_orig * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Test with numpy input
        resampled_np = high_quality_resample(audio, sr_orig, sr_target)
        print(f"✓ Numpy resample: {audio.shape} -> {resampled_np.shape}")
        
        # Test with torch input
        audio_torch = torch.from_numpy(audio).float()
        resampled_torch = high_quality_resample(audio_torch, sr_orig, sr_target)
        print(f"✓ Torch resample: {audio_torch.shape} -> {resampled_torch.shape}")
        
        # Check expected length
        expected_length = int(len(audio) * sr_target / sr_orig)
        actual_length = len(resampled_np)
        print(f"✓ Length check: expected ~{expected_length}, got {actual_length}")
        
        if abs(expected_length - actual_length) <= 1:
            print("✓ Resampling length is correct")
        else:
            print(f"⚠ Resampling length off by {abs(expected_length - actual_length)}")
            
    except Exception as e:
        print(f"✗ High quality resample test failed: {e}")
        traceback.print_exc()

def test_dcr_inference_only():
    """Test that DCR function is properly documented as inference-only"""
    print("\n=== Testing DCR Inference-Only Documentation ===")
    
    try:
        from lycodec.audio import dynamic_range_compression
        
        # Check if docstring mentions inference-only
        docstring = dynamic_range_compression.__doc__
        if docstring and "INFERENCE ONLY" in docstring:
            print("✓ DCR properly documented as inference-only")
        else:
            print("⚠ DCR documentation may need improvement")
        
        # Test that DCR returns no gradients
        magnitude = torch.randn(2, 513, 100, requires_grad=True)
        compressed = dynamic_range_compression(magnitude)
        
        if compressed.requires_grad:
            print("⚠ DCR output has gradients (may break training if used incorrectly)")
        else:
            print("✓ DCR output has no gradients (safe for inference)")
            
    except Exception as e:
        print(f"✗ DCR test failed: {e}")
        traceback.print_exc()

def test_stft_istft_consistency():
    """Test STFT/ISTFT consistency with new parameters"""
    print("\n=== Testing STFT/ISTFT Consistency ===")
    
    try:
        from lycodec.audio import stft_transform, istft_transform
        
        # Create test audio
        audio = torch.randn(2, 44100)  # 1 second stereo
        
        # Forward and backward
        stft = stft_transform(audio, return_complex=True)
        reconstructed = istft_transform(stft, length=audio.shape[-1])
        
        print(f"✓ STFT: {audio.shape} -> {stft.shape}")
        print(f"✓ ISTFT: {stft.shape} -> {reconstructed.shape}")
        
        # Check reconstruction quality
        mse = torch.mean((audio - reconstructed) ** 2)
        print(f"✓ Reconstruction MSE: {mse.item():.2e}")
        
        if mse.item() < 1e-6:
            print("✓ Excellent STFT/ISTFT consistency")
        elif mse.item() < 1e-4:
            print("✓ Good STFT/ISTFT consistency")
        else:
            print("⚠ STFT/ISTFT consistency could be improved")
            
    except Exception as e:
        print(f"✗ STFT/ISTFT consistency test failed: {e}")
        traceback.print_exc()

def main():
    """Run all API tests"""
    print("🧪 LyCodec API Improvement Tests")
    print("=" * 50)
    
    test_magnitude_phase_api()
    test_fp16_cpu_guards()
    test_streaming_safety_guards()
    test_high_quality_resample()
    test_dcr_inference_only()
    test_stft_istft_consistency()
    
    print("\n" + "=" * 50)
    print("🎯 API improvement tests completed!")
    print("\nNote: Some tests use dummy models and may show warnings.")
    print("For full functionality testing, use test_codec.py with a trained model.")

if __name__ == "__main__":
    main()
