#!/usr/bin/env python3
"""
Compression ratio analysis for LyCodec
Measures true compression including file I/O, headers, and metadata
"""

import torch
import numpy as np
import tempfile
import os
import gzip
import pickle
from pathlib import Path
import soundfile as sf

def measure_file_compression_ratio(codec, audio_path, output_dir=None):
    """
    Measure true compression ratio including file I/O overhead
    
    Args:
        codec: LyCodec instance
        audio_path: Path to input audio file
        output_dir: Directory for output files (temp if None)
    
    Returns:
        dict with compression statistics
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    else:
        Path(output_dir).mkdir(exist_ok=True)
    
    # Get input file size
    input_size = os.path.getsize(audio_path)
    
    # Load and get raw audio size
    audio, sr = sf.read(audio_path, always_2d=True)
    audio = audio.T  # [channels, samples]
    raw_audio_size = audio.nbytes
    
    print(f"Input file: {input_size:,} bytes")
    print(f"Raw audio: {raw_audio_size:,} bytes ({audio.shape})")
    
    # Encode to latent
    latent = codec.encode(audio)
    
    # Save latent in different formats and measure sizes
    results = {
        'input_file_size': input_size,
        'raw_audio_size': raw_audio_size,
        'audio_shape': audio.shape,
        'latent_shape': latent.shape if hasattr(latent, 'shape') else str(type(latent)),
        'formats': {}
    }
    
    # 1. PyTorch tensor (.pth)
    pth_path = Path(output_dir) / "latent.pth"
    torch.save(latent, pth_path)
    pth_size = pth_path.stat().st_size
    results['formats']['pytorch'] = {
        'size': pth_size,
        'ratio_vs_file': input_size / pth_size,
        'ratio_vs_raw': raw_audio_size / pth_size
    }
    
    # 2. Compressed PyTorch (.pth.gz)
    with open(pth_path, 'rb') as f_in:
        with gzip.open(str(pth_path) + '.gz', 'wb') as f_out:
            f_out.writelines(f_in)
    pth_gz_size = (pth_path.parent / (pth_path.name + '.gz')).stat().st_size
    results['formats']['pytorch_gzip'] = {
        'size': pth_gz_size,
        'ratio_vs_file': input_size / pth_gz_size,
        'ratio_vs_raw': raw_audio_size / pth_gz_size
    }
    
    # 3. Numpy compressed (.npz)
    if hasattr(latent, 'numpy'):
        latent_np = latent.cpu().numpy()
    else:
        latent_np = np.array(latent)
    
    npz_path = Path(output_dir) / "latent.npz"
    np.savez_compressed(npz_path, latent=latent_np)
    npz_size = npz_path.stat().st_size
    results['formats']['numpy_compressed'] = {
        'size': npz_size,
        'ratio_vs_file': input_size / npz_size,
        'ratio_vs_raw': raw_audio_size / npz_size
    }
    
    # 4. Pickle compressed
    pkl_path = Path(output_dir) / "latent.pkl.gz"
    with gzip.open(pkl_path, 'wb') as f:
        pickle.dump(latent_np, f, protocol=pickle.HIGHEST_PROTOCOL)
    pkl_size = pkl_path.stat().st_size
    results['formats']['pickle_gzip'] = {
        'size': pkl_size,
        'ratio_vs_file': input_size / pkl_size,
        'ratio_vs_raw': raw_audio_size / pkl_size
    }
    
    # 5. Raw float32 binary (theoretical minimum)
    raw_path = Path(output_dir) / "latent.raw"
    latent_np.astype(np.float32).tofile(raw_path)
    raw_size = raw_path.stat().st_size
    results['formats']['raw_binary'] = {
        'size': raw_size,
        'ratio_vs_file': input_size / raw_size,
        'ratio_vs_raw': raw_audio_size / raw_size
    }
    
    # Find best compression
    best_format = max(results['formats'].items(), 
                     key=lambda x: x[1]['ratio_vs_raw'])
    results['best_format'] = {
        'name': best_format[0],
        'compression_ratio': best_format[1]['ratio_vs_raw']
    }
    
    return results

def analyze_compression_by_duration(codec, durations=[1, 5, 10, 30]):
    """
    Analyze how compression ratio varies with audio duration
    """
    print("\n=== Compression vs Duration Analysis ===")
    
    results = []
    
    for duration in durations:
        # Generate test audio
        sr = 44100
        samples = int(sr * duration)
        
        # Stereo sine wave test signal
        t = np.linspace(0, duration, samples)
        audio = np.array([
            np.sin(2 * np.pi * 440 * t),  # 440 Hz left
            np.sin(2 * np.pi * 880 * t)   # 880 Hz right
        ])
        
        # Save as temporary WAV
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio.T, sr)
            temp_wav = f.name
        
        try:
            # Measure compression
            result = measure_file_compression_ratio(codec, temp_wav)
            result['duration'] = duration
            results.append(result)
            
            best = result['best_format']
            print(f"{duration:2d}s: {best['compression_ratio']:.1f}x compression ({best['name']})")
            
        finally:
            os.unlink(temp_wav)
    
    return results

def print_compression_report(results):
    """Print detailed compression analysis report"""
    print(f"\n=== Compression Report ===")
    print(f"Audio: {results['audio_shape']} -> Latent: {results['latent_shape']}")
    print(f"Input file: {results['input_file_size']:,} bytes")
    print(f"Raw audio: {results['raw_audio_size']:,} bytes")
    print()
    
    print("Format Comparison:")
    print(f"{'Format':<20} {'Size':<12} {'vs File':<10} {'vs Raw':<10}")
    print("-" * 52)
    
    for fmt_name, fmt_data in results['formats'].items():
        print(f"{fmt_name:<20} {fmt_data['size']:>8,} {fmt_data['ratio_vs_file']:>8.1f}x {fmt_data['ratio_vs_raw']:>8.1f}x")
    
    print()
    print(f"🏆 Best: {results['best_format']['name']} "
          f"({results['best_format']['compression_ratio']:.1f}x compression)")

def main():
    """
    Main compression analysis
    Run this with a trained model for real analysis
    """
    try:
        from lycodec import LyCodec
        
        # Create codec (will use dummy weights if no model provided)
        codec = LyCodec()
        print("Created LyCodec instance for compression testing")
        
        # Analyze compression vs duration
        duration_results = analyze_compression_by_duration(codec)
        
        # Show theoretical f10c10 compression
        print(f"\n=== Theoretical vs Actual ===")
        print(f"Target: f10c10 = 100x compression")
        
        for result in duration_results:
            actual = result['best_format']['compression_ratio']
            theoretical = 100.0
            efficiency = actual / theoretical * 100
            print(f"{result['duration']:2d}s: {actual:.1f}x actual "
                  f"({efficiency:.1f}% of theoretical)")
        
        print(f"\nNote: Real compression with trained models may differ significantly")
        print(f"This measures format overhead and storage efficiency")
        
    except ImportError:
        print("LyCodec not available - run this after training a model")
    except Exception as e:
        print(f"Error in compression analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
