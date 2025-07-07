#!/usr/bin/env python3
"""
Accelerate 설정 직접 수정 - DDP find_unused_parameters 추가
"""

import os
import yaml
from pathlib import Path

def fix_accelerate_config():
    """기존 Accelerate 설정에 ddp_kwargs 추가"""
    
    config_path = Path.home() / ".cache/huggingface/accelerate/default_config.yaml"
    
    print(f"🔧 Fixing Accelerate config at: {config_path}")
    
    if not config_path.exists():
        print("❌ Config file not found!")
        return False
    
    # 현재 설정 읽기
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("📋 Current config:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # 백업 생성
    backup_path = config_path.with_suffix('.yaml.backup')
    with open(backup_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"💾 Backup saved: {backup_path}")
    
    # ddp_kwargs 추가
    config['ddp_kwargs'] = {
        'find_unused_parameters': True,
        'broadcast_buffers': True,
        'bucket_cap_mb': 25
    }
    
    # mixed_precision을 no로 변경 (안정성을 위해)
    config['mixed_precision'] = 'no'
    
    # 수정된 설정 저장
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("\n✅ Fixed config:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    return True

def create_emergency_accelerate_config():
    """완전히 새로운 안전한 설정 생성"""
    
    config = {
        'compute_environment': 'LOCAL_MACHINE',
        'distributed_type': 'MULTI_GPU',
        'downcast_bf16': 'no',
        'gpu_ids': 'all',
        'machine_rank': 0,
        'main_training_function': 'main',
        'mixed_precision': 'no',  # FP16 비활성화로 안정성 확보
        'num_machines': 1,
        'num_processes': 4,
        'rdzv_backend': 'static',
        'same_network': True,
        'tpu_env': [],
        'tpu_use_cluster': False,
        'tpu_use_sudo': False,
        'use_cpu': False,
        'enable_cpu_affinity': False,
        'debug': False,
        'ddp_kwargs': {
            'find_unused_parameters': True,
            'broadcast_buffers': True,
            'bucket_cap_mb': 25
        }
    }
    
    config_path = Path.home() / ".cache/huggingface/accelerate/default_config.yaml"
    
    # 원본 백업
    if config_path.exists():
        backup_path = config_path.with_suffix('.yaml.original')
        config_path.rename(backup_path)
        print(f"💾 Original config backed up: {backup_path}")
    
    # 새 설정 저장
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Created new safe Accelerate config")
    return True

def verify_config():
    """설정 확인"""
    config_path = Path.home() / ".cache/huggingface/accelerate/default_config.yaml"
    
    if not config_path.exists():
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n🔍 Verification:")
    has_ddp = 'ddp_kwargs' in config
    has_unused_params = has_ddp and config['ddp_kwargs'].get('find_unused_parameters', False)
    
    print(f"   ✅ ddp_kwargs present: {has_ddp}")
    print(f"   ✅ find_unused_parameters: {has_unused_params}")
    print(f"   ✅ mixed_precision: {config.get('mixed_precision', 'unknown')}")
    
    return has_ddp and has_unused_params

if __name__ == '__main__':
    print("🚨 DDP Configuration Fix")
    print("=" * 40)
    
    # 1단계: 기존 설정 수정 시도
    success = fix_accelerate_config()
    
    if not success:
        print("\n🔄 Creating new configuration...")
        success = create_emergency_accelerate_config()
    
    # 2단계: 설정 확인
    if verify_config():
        print("\n🎉 SUCCESS! Configuration fixed.")
        print("\n🚀 Now run:")
        print("   accelerate launch train.py --config config_emergency.yaml")
    else:
        print("\n❌ Configuration fix failed.")
        print("\n🚀 Try single GPU instead:")
        print("   export CUDA_VISIBLE_DEVICES=0")
        print("   accelerate launch --num_processes=1 train.py --config config_emergency.yaml")