#!/usr/bin/env python3
"""
LyCodec Streaming Demo
Demonstrates real-time audio streaming with ~100ms latency
"""

import torch
import numpy as np
import soundfile as sf
import time
from pathlib import Path
import argparse

from lycodec import create_streaming_decoder

def simulate_latent_stream(audio_file: str, chunk_duration: float = 0.1):
    """
    Simulate a stream of latent chunks from an audio file
    In practice, these would come from a network stream or live encoding
    """
    print(f"Loading audio: {audio_file}")
    
    # Load and encode the full audio first (simulation)
    from lycodec import LyCodec
    codec = LyCodec(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Encode full audio to get latent representation
    latent = codec.encode(audio_file)
    
    if isinstance(latent, dict):
        # Handle chunked encoding
        full_latent = latent['latents']
    else:
        full_latent = latent
    
    # Calculate chunk size in latent space
    sample_rate = 44100
    original_duration = full_latent.shape[-1] * 10 * 512 / sample_rate  # Reverse f10c10
    chunks_per_second = 1.0 / chunk_duration
    latent_chunk_size = max(1, int(full_latent.shape[-1] / (original_duration * chunks_per_second)))
    
    print(f"Simulating {original_duration:.1f}s audio as latent stream")
    print(f"Latent chunks: {latent_chunk_size} frames per {chunk_duration*1000:.0f}ms")
    
    # Stream latent chunks
    start_pos = 0
    while start_pos < full_latent.shape[-1]:
        end_pos = min(start_pos + latent_chunk_size, full_latent.shape[-1])
        
        chunk = full_latent[..., start_pos:end_pos]
        if chunk.shape[-1] > 0:  # Only yield non-empty chunks
            yield chunk
        
        start_pos = end_pos
        
        # Simulate real-time streaming delay
        time.sleep(chunk_duration * 0.9)  # Slightly faster to build up buffer

def demo_streaming_decoder(audio_file: str, output_file: str, latency_ms: float = 100):
    """Demonstrate streaming decoding with specified latency"""
    
    print("🎵 LyCodec Streaming Decoder Demo")
    print(f"Target latency: {latency_ms}ms")
    
    # Create streaming decoder
    streaming_decoder = create_streaming_decoder(
        model_path=None,  # Will use default/untrained model for demo
        latency_ms=latency_ms
    )
    
    print(f"Actual latency: {streaming_decoder.get_latency_ms():.1f}ms")
    
    # Simulate latent stream
    chunk_duration = latency_ms / 1000  # Convert to seconds
    latent_stream = simulate_latent_stream(audio_file, chunk_duration)
    
    # Collect decoded audio chunks
    decoded_chunks = []
    chunk_count = 0
    total_decode_time = 0
    
    print("\n🚀 Starting streaming decode...")
    
    with streaming_decoder.streaming_session():
        for audio_chunk in streaming_decoder.stream_decode_generator(latent_stream):
            decoded_chunks.append(audio_chunk)
            chunk_count += 1
            
            # Monitor performance
            stats = streaming_decoder.get_stats()
            if chunk_count % 10 == 0:  # Log every 10 chunks
                print(f"Processed {chunk_count} chunks, "
                      f"Queue: {stats['input_queue_size']}/{stats['output_queue_size']}")
    
    if decoded_chunks:
        # Concatenate all chunks
        full_audio = np.concatenate(decoded_chunks, axis=1)  # [2, T]
        
        # Save output
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        # Transpose for soundfile: [T, 2]
        sf.write(output_path, full_audio.T, 44100)
        
        print(f"\n✅ Streaming decode completed!")
        print(f"Output saved: {output_path}")
        print(f"Total chunks: {chunk_count}")
        print(f"Output duration: {full_audio.shape[1]/44100:.2f}s")
    else:
        print("❌ No audio chunks decoded")

def demo_real_time_processing():
    """Demo real-time processing capabilities"""
    print("\n🔥 Real-time Processing Demo")
    
    # Create decoder with minimal latency
    decoder = create_streaming_decoder(model_path=None, latency_ms=50)
    
    print(f"Ultra-low latency: {decoder.get_latency_ms():.1f}ms")
    
    # Generate test latent (would come from live encoder in practice)
    test_latent = torch.randn(1, 64, 8, 16)  # Small test chunk
    
    with decoder.streaming_session():
        # Measure processing time
        start_time = time.time()
        
        success = decoder.put_latent_chunk(test_latent)
        if success:
            chunk_info = decoder.get_audio_chunk(timeout=1.0)
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            
            if chunk_info:
                print(f"✅ Processing time: {processing_time:.1f}ms")
                print(f"   Decode time: {chunk_info['decode_time_ms']:.1f}ms")
                print(f"   Audio shape: {chunk_info['audio'].shape}")
                print(f"   Real-time factor: {processing_time/decoder.get_latency_ms():.2f}x")
            else:
                print("❌ No output received")
        else:
            print("❌ Failed to submit chunk")

def main():
    parser = argparse.ArgumentParser(description='LyCodec Streaming Demo')
    parser.add_argument('--input', type=str, default='test_audio.wav',
                       help='Input audio file')
    parser.add_argument('--output', type=str, default='output/streamed_audio.wav',
                       help='Output audio file')
    parser.add_argument('--latency', type=float, default=100,
                       help='Target latency in milliseconds')
    parser.add_argument('--realtime-test', action='store_true',
                       help='Run real-time processing test')
    
    args = parser.parse_args()
    
    if args.realtime_test:
        demo_real_time_processing()
    else:
        if Path(args.input).exists():
            demo_streaming_decoder(args.input, args.output, args.latency)
        else:
            print(f"❌ Input file not found: {args.input}")
            print("Run with --realtime-test for synthetic demo")

if __name__ == "__main__":
    main()
