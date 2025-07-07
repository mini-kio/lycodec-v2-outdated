#!/usr/bin/env python3
"""
Minimal test to isolate the model forward pass issue
"""

import torch
import yaml
from lycodec.models import LyCodecModel
from lycodec.audio import to_complex_spec, to_magnitude_phase

def test_model_forward():
    """Test basic model forward pass"""
    print("🔍 Loading config...")
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("🔍 Creating model...")
    model_config = config.get('model', {})
    model_config['use_triton'] = False
    model = LyCodecModel(**model_config)
    
    if torch.cuda.is_available():
        print(f"🔍 Moving model to CUDA...")
        model = model.cuda()
    
    model.eval()
    
    print("🔍 Creating test input...")
    # Create dummy stereo audio: [batch=1, channels=2, time=220500]
    dummy_audio = torch.randn(1, 2, 220500)
    if torch.cuda.is_available():
        dummy_audio = dummy_audio.cuda()
    
    print("🔍 Converting to complex spectrogram...")
    complex_specs = []
    magnitude_specs = []
    
    for i in range(2):  # Stereo channels
        complex_spec = to_complex_spec(dummy_audio[:, i])
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
    
    print(f"🔍 Input shape: {target_complex_input.shape}")
    print(f"🔍 Magnitude shape: {magnitude_input.shape}")
    
    print("🔍 Testing model forward pass...")
    with torch.no_grad():
        try:
            pred_real, pred_imag, pred_latent = model(target_complex_input, magnitude_input)
            print(f"✅ Model forward completed!")
            print(f"   pred_real shape: {pred_real.shape}")
            print(f"   pred_imag shape: {pred_imag.shape}")
            print(f"   pred_latent shape: {pred_latent.shape}")
        except Exception as e:
            print(f"❌ Model forward failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    test_model_forward()
