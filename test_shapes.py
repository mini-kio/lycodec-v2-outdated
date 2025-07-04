"""
LyCodec Shape Test Script
=========================

테스트 스크립트로 모델의 입력/출력 shape을 검증하고 
shape 관련 오류를 사전에 발견합니다.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import traceback

# LyCodec 컴포넌트 import
from lycodec import (
    LyCodecModel, LyCodecConfig,
    SAMPLE_RATE, CHANNELS, SEGMENT_LENGTH, SEGMENT_SAMPLES
)


def test_model_shapes():
    """모델의 기본 shape 테스트"""
    print("=" * 60)
    print("LyCodec 모델 Shape 테스트 시작")
    print("=" * 60)
    
    # 테스트 설정
    batch_size = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"디바이스: {device}")
    print(f"배치 크기: {batch_size}")
    print(f"오디오 설정: {SAMPLE_RATE}Hz, {CHANNELS}채널, {SEGMENT_LENGTH}초")
    print(f"세그먼트 샘플: {SEGMENT_SAMPLES}")
    print()
    
    try:
        # 모델 설정 및 초기화
        config = LyCodecConfig()
        print(f"모델 설정:")
        print(f"  - Hidden dim: {config.hidden_dim}")
        print(f"  - Layers: {config.num_layers}")
        print(f"  - Attention heads: {config.num_attention_heads}")
        print(f"  - Harmonics: {config.harmonics_count}")
        print()
        
        # 모델 생성
        model = LyCodecModel(config).to(device)
        model.eval()
        
        print(f"모델 파라미터 수: {model.get_model_size():,}")
        print(f"모델 메모리 사용량: {model.get_memory_usage()}")
        print()
        
        # 테스트 오디오 생성
        print("테스트 오디오 생성...")
        audio_input = torch.randn(batch_size, CHANNELS, SEGMENT_SAMPLES).to(device)
        print(f"입력 오디오 shape: {audio_input.shape}")
        print(f"입력 오디오 dtype: {audio_input.dtype}")
        print(f"입력 오디오 범위: [{audio_input.min().item():.3f}, {audio_input.max().item():.3f}]")
        print()
        
        # 모델 forward 테스트
        print("모델 forward 테스트...")
        with torch.no_grad():
            output = model(audio_input, training=False)
        
        print("Forward 성공!")
        print()
        
        # 출력 shape 검증
        print("출력 검증:")
        reconstructed_audio = output['reconstructed_audio']
        print(f"재구성 오디오 shape: {reconstructed_audio.shape}")
        print(f"재구성 오디오 dtype: {reconstructed_audio.dtype}")
        print(f"재구성 오디오 범위: [{reconstructed_audio.min().item():.3f}, {reconstructed_audio.max().item():.3f}]")
        
        # Shape 일치 확인
        if reconstructed_audio.shape == audio_input.shape:
            print("✅ 입력과 출력 shape 일치!")
        else:
            print(f"❌ Shape 불일치! 입력: {audio_input.shape}, 출력: {reconstructed_audio.shape}")
        
        print()
        
        # 기타 출력 확인
        print("기타 출력:")
        for key, value in output.items():
            if key != 'reconstructed_audio':
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: {value.shape} ({value.dtype})")
                elif isinstance(value, dict):
                    print(f"  - {key}: Dict with {len(value)} keys")
                    for k, v in value.items():
                        if isinstance(v, torch.Tensor):
                            print(f"    - {k}: {v.shape}")
                        else:
                            print(f"    - {k}: {type(v)}")
                else:
                    print(f"  - {key}: {type(value)}")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print(f"상세 오류:")
        traceback.print_exc()
        return False


def test_encoder_decoder_shapes():
    """Encoder와 Decoder의 개별 shape 테스트"""
    print("=" * 60)
    print("Encoder/Decoder Shape 테스트")
    print("=" * 60)
    
    batch_size = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        config = LyCodecConfig()
        model = LyCodecModel(config).to(device)
        model.eval()
        
        # 테스트 오디오
        audio_input = torch.randn(batch_size, CHANNELS, SEGMENT_SAMPLES).to(device)
        print(f"입력 오디오: {audio_input.shape}")
        
        with torch.no_grad():
            # Encoder 테스트
            print("\n1. Encoder 테스트...")
            latent_features, encoder_metadata = model.encoder(audio_input)
            print(f"  Latent features: {latent_features.shape}")
            print(f"  Encoder metadata keys: {list(encoder_metadata.keys())}")
            
            # Quantizer 테스트
            print("\n2. Quantizer 테스트...")
            quantized_features, quant_loss, quant_metadata = model.quantizer(
                latent_features, training=False
            )
            print(f"  Quantized features: {quantized_features.shape}")
            print(f"  Quantization loss: {quant_loss.item():.6f}")
            print(f"  Quantizer metadata keys: {list(quant_metadata.keys())}")
            
            # Decoder 테스트
            print("\n3. Decoder 테스트...")
            decoder_metadata = {**encoder_metadata, **quant_metadata}
            reconstructed = model.decoder(quantized_features, decoder_metadata)
            print(f"  Reconstructed audio: {reconstructed.shape}")
            
            # Shape 검증
            if reconstructed.shape == audio_input.shape:
                print("✅ 전체 파이프라인 shape 일치!")
            else:
                print(f"❌ Shape 불일치! 입력: {audio_input.shape}, 출력: {reconstructed.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        traceback.print_exc()
        return False


def test_different_batch_sizes():
    """다양한 배치 크기로 테스트"""
    print("=" * 60)
    print("다양한 배치 크기 테스트")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_sizes = [1, 2, 4, 8]
    
    try:
        config = LyCodecConfig()
        model = LyCodecModel(config).to(device)
        model.eval()
        
        for batch_size in batch_sizes:
            print(f"\n배치 크기 {batch_size} 테스트...")
            
            # 메모리 사용량 확인
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated(device) / 1024**2  # MB
            
            audio_input = torch.randn(batch_size, CHANNELS, SEGMENT_SAMPLES).to(device)
            
            with torch.no_grad():
                output = model(audio_input, training=False)
            
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated(device) / 1024**2  # MB
                memory_used = memory_after - memory_before
                print(f"  입력 shape: {audio_input.shape}")
                print(f"  출력 shape: {output['reconstructed_audio'].shape}")
                print(f"  메모리 사용량: {memory_used:.1f} MB")
            else:
                print(f"  입력 shape: {audio_input.shape}")
                print(f"  출력 shape: {output['reconstructed_audio'].shape}")
            
            # Shape 검증
            if output['reconstructed_audio'].shape == audio_input.shape:
                print(f"  ✅ 배치 크기 {batch_size} 성공!")
            else:
                print(f"  ❌ 배치 크기 {batch_size} 실패!")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        traceback.print_exc()
        return False


def test_loss_function_shapes():
    """Loss 함수의 shape 테스트"""
    print("=" * 60)
    print("Loss 함수 Shape 테스트")
    print("=" * 60)
    
    from train import LyCodecLoss
    
    batch_size = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Loss 함수 설정
        loss_config = {
            'reconstruction_weight': 1.0,
            'quantization_weight': 0.25,
            'perceptual_weight': 0.1,
            'bitrate_weight': 0.01
        }
        loss_fn = LyCodecLoss(loss_config)
        
        # 모델 출력 시뮬레이션
        config = LyCodecConfig()
        model = LyCodecModel(config).to(device)
        model.eval()
        
        audio_input = torch.randn(batch_size, CHANNELS, SEGMENT_SAMPLES).to(device)
        
        with torch.no_grad():
            model_output = model(audio_input, training=False)
        
        print(f"모델 출력 keys: {list(model_output.keys())}")
        
        # Loss 계산
        loss_dict = loss_fn(model_output, audio_input)
        
        print("\nLoss 계산 결과:")
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: {value.item():.6f} (shape: {value.shape})")
            else:
                print(f"  - {key}: {value}")
        
        # Loss 값 검증
        total_loss = loss_dict['total_loss']
        if total_loss.shape == torch.Size([]):  # Scalar
            print("✅ Loss shape 정상!")
        else:
            print(f"❌ Loss shape 비정상: {total_loss.shape}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        traceback.print_exc()
        return False


def test_gradient_flow():
    """Gradient flow 테스트"""
    print("=" * 60)
    print("Gradient Flow 테스트")
    print("=" * 60)
    
    from train import LyCodecLoss
    
    batch_size = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        config = LyCodecConfig()
        model = LyCodecModel(config).to(device)
        model.train()  # Training mode
        
        # Loss 함수
        loss_config = {
            'reconstruction_weight': 1.0,
            'quantization_weight': 0.25,
            'perceptual_weight': 0.1,
            'bitrate_weight': 0.01
        }
        loss_fn = LyCodecLoss(loss_config)
        
        # 옵티마이저
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Forward pass
        audio_input = torch.randn(batch_size, CHANNELS, SEGMENT_SAMPLES).to(device)
        model_output = model(audio_input, training=True)
        loss_dict = loss_fn(model_output, audio_input)
        total_loss = loss_dict['total_loss']
        
        print(f"Total loss: {total_loss.item():.6f}")
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient 확인
        grad_norms = []
        param_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                param_count += 1
        
        print(f"파라미터 수 (gradient 있음): {param_count}")
        print(f"Gradient norm 범위: [{min(grad_norms):.6f}, {max(grad_norms):.6f}]")
        print(f"평균 gradient norm: {np.mean(grad_norms):.6f}")
        
        # Optimizer step
        optimizer.step()
        
        print("✅ Gradient flow 정상!")
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        traceback.print_exc()
        return False


def main():
    """메인 테스트 함수"""
    print("LyCodec Shape 테스트 시작\n")
    
    tests = [
        ("기본 모델 Shape", test_model_shapes),
        ("Encoder/Decoder Shape", test_encoder_decoder_shapes),
        ("다양한 배치 크기", test_different_batch_sizes),
        ("Loss 함수 Shape", test_loss_function_shapes),
        ("Gradient Flow", test_gradient_flow),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name} 테스트 중...")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                print(f"✅ {test_name} 성공!")
            else:
                print(f"❌ {test_name} 실패!")
        except Exception as e:
            print(f"❌ {test_name} 오류: {e}")
            results[test_name] = False
        
        print("-" * 60)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for test_name, result in results.items():
        status = "✅ 성공" if result else "❌ 실패"
        print(f"{test_name}: {status}")
    
    print(f"\n전체 결과: {success_count}/{total_count} 성공")
    
    if success_count == total_count:
        print("🎉 모든 테스트 성공!")
        return True
    else:
        print("⚠️  일부 테스트 실패")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
