"""
LyCodec v2.1 Main Model Architecture - Fixed Linear Attention Only
=================================================================

Production-ready audio codec architecture with enhanced psychoacoustic modeling
and DDSP vocoder synthesis. Optimized for 44.1kHz stereo with ~45M parameters
targeting V100×4 16GB training and <8GB inference. Uses exclusively Linear Attention.
Fixed dimension handling and einsum issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math

from .psychoacoustic import PsychoacousticTransform, FastRMSNorm2D
from .vocoder import DDSPVocoder
from .utils import ProductionAdaptiveBitAllocator


@dataclass
class LyCodecConfig:
    """
    Configuration for LyCodec v2.1 production model.
    
    Architecture optimized for 44.1kHz stereo with memory-efficient design
    targeting ~45M parameters for V100×4 16GB training constraint.
    """
    # Audio specifications
    sample_rate: int = 44100
    channels: int = 2
    segment_length: float = 5.0
    segment_samples: int = 220500  # 44100 * 5.0
    
    # Model architecture dimensions
    hidden_dim: int = 512
    num_layers: int = 8
    num_attention_heads: int = 8
    feedforward_dim: int = 2048
    
    # Psychoacoustic transform configuration
    psycho_bands: int = 64
    psycho_window_size: int = 2048
    psycho_hop_length: int = 512
    

    # Frontend configuration
    frontend_stride: int = 32
    frontend_kernel_size: int = 64
    
    # Low-rank fusion parameters
    low_rank_dim: int = 64
    fusion_layers: List[int] = None
    residual_weight_init: float = 0.1
    
    # DDSP vocoder configuration
    harmonics_count: int = 48
    noise_bands: int = 8
    vocoder_hidden_dim: int = 256
    
    # Bit allocation parameters
    min_bitrate: int = 128
    max_bitrate: int = 320
    complexity_lookahead: int = 16
    safety_factor: float = 1.2
    
    def __post_init__(self):
        if self.fusion_layers is None:
            self.fusion_layers = [2, 4, 6]  # Cross-level fusion at these layers


class LowRankLinearFusion(nn.Module):
    """
    Memory-efficient low-rank linear transformation with learnable residual weighting.
    
    Implements O(N²D) → O(NrD) complexity reduction through SVD-based decomposition
    with dynamic residual control for cross-level information flow optimization.
    """
    
    def __init__(self, input_dim: int, output_dim: int, rank: int = 64, 
                 residual_weight_init: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        
        # Low-rank decomposition: W = U @ V^T
        self.U = nn.Parameter(torch.randn(input_dim, rank) / math.sqrt(rank))
        self.V = nn.Parameter(torch.randn(rank, output_dim) / math.sqrt(rank))
        
        # Learnable residual weight with sigmoid gating
        # Transform to sigmoid space for stable gradients
        self.residual_weight_logit = nn.Parameter(
            torch.logit(torch.tensor(residual_weight_init))
        )
        
        # Optional bias term
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize low-rank matrices with orthogonal structure."""
        # Initialize U with left singular vectors
        with torch.no_grad():
            q, _ = torch.linalg.qr(self.U)
            self.U.copy_(q)
            
        # Initialize V with scaled random orthogonal
        with torch.no_grad():
            q, _ = torch.linalg.qr(self.V.T)
            self.V.copy_(q.T * math.sqrt(self.rank))
    
    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional residual connection.
        
        Args:
            x: Input tensor [..., input_dim]
            residual: Optional residual tensor [..., output_dim]
            
        Returns:
            Output tensor [..., output_dim] with residual fusion
        """
        # Low-rank transformation: x @ U @ V^T + bias
        low_rank_output = torch.matmul(torch.matmul(x, self.U), self.V) + self.bias
        
        if residual is not None:
            # Sigmoid-gated residual weighting
            alpha = torch.sigmoid(self.residual_weight_logit)
            output = (1 - alpha) * low_rank_output + alpha * residual
        else:
            output = low_rank_output
            
        return output
    
    @property
    def effective_residual_weight(self) -> float:
        """Get current effective residual weight."""
        return torch.sigmoid(self.residual_weight_logit).item()


class BottleneckDown(nn.Module):
    """Temporal and channel downsampling to create compact latents."""

    def __init__(self, hidden_dim: int, factor: int = 10):
        super().__init__()
        self.factor = factor
        reduced_dim = hidden_dim // factor
        self.conv = nn.Conv1d(hidden_dim, reduced_dim, kernel_size=1, stride=factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, hidden_dim]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class BottleneckUp(nn.Module):
    """Re-expand compact latents back to encoder dimensionality."""

    def __init__(self, hidden_dim: int, factor: int = 10):
        super().__init__()
        self.factor = factor
        reduced_dim = hidden_dim // factor
        self.deconv = nn.ConvTranspose1d(reduced_dim, hidden_dim, kernel_size=1, stride=factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, hidden_dim // factor]
        x = x.transpose(1, 2)
        x = self.deconv(x)
        x = x.transpose(1, 2)
        return x


class LyCodecTransformerLayer(nn.Module):
    """
    Enhanced transformer layer with psychoacoustic Linear Attention and low-rank fusion.
    
    Integrates orthogonal attention matrices, FastRMSNorm2D, and cross-level
    information flow using exclusively Linear Attention for production-grade audio modeling.
    """
    
    def __init__(self, config: LyCodecConfig, layer_idx: int):
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_dim // self.num_heads
        
        # Enhanced psychoacoustic attention with orthogonal initialization (Linear Attention only)
        self.psycho_attention = PsychoacousticTransform(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_attention_heads,
            psycho_bands=config.psycho_bands,
            sample_rate=config.sample_rate
        )
        
        # FastRMSNorm2D with cross-platform compatibility
        self.attention_norm = FastRMSNorm2D(config.hidden_dim)
        self.ffn_norm = FastRMSNorm2D(config.hidden_dim)
        
        # Low-rank feedforward network
        self.ffn = nn.Sequential(
            LowRankLinearFusion(
                config.hidden_dim, 
                config.feedforward_dim, 
                rank=config.low_rank_dim
            ),
            nn.GELU(),
            LowRankLinearFusion(
                config.feedforward_dim, 
                config.hidden_dim, 
                rank=config.low_rank_dim
            )
        )
        
        # Cross-level fusion for information flow optimization
        self.has_fusion = layer_idx in config.fusion_layers
        if self.has_fusion:
            self.cross_level_fusion = LowRankLinearFusion(
                config.hidden_dim * 2,  # Concatenated features
                config.hidden_dim,
                rank=config.low_rank_dim,
                residual_weight_init=config.residual_weight_init
            )
    
    def forward(self, x: torch.Tensor, cross_level_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional cross-level feature fusion using Linear Attention.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            cross_level_features: Optional features from other layers
            
        Returns:
            Transformed tensor with same shape as input
        """
        # Pre-norm psychoacoustic attention with residual connection (Linear Attention only)
        normed_x = self.attention_norm(x)
        attn_output = self.psycho_attention(normed_x)
        x = x + attn_output
        
        # Cross-level feature fusion if applicable
        if self.has_fusion and cross_level_features is not None:
            # Concatenate and fuse cross-level information
            fused_features = torch.cat([x, cross_level_features], dim=-1)
            x = self.cross_level_fusion(fused_features, residual=x)
        
        # Pre-norm feedforward with residual connection
        normed_x = self.ffn_norm(x)
        ffn_output = self.ffn(normed_x)
        x = x + ffn_output
        
        return x
    
    def get_attention_info(self) -> Dict[str, Union[str, int, List[str]]]:
        """
        Get information about the Linear Attention mechanism used.
        
        Returns:
            Dictionary with attention configuration details
        """
        return {
            'attention_type': 'Linear Attention (Psychoacoustic)',
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'psycho_bands': self.config.psycho_bands,
            'orthogonal_projections': True,
            'cross_level_fusion': self.has_fusion,
            'computational_complexity': 'O(N × D)',  # Linear in sequence length
            'memory_complexity': 'O(N × D)',
            'advantages': [
                'Linear scaling with sequence length',
                'Psychoacoustic masking integration',
                'Orthogonal weight initialization',
                'Cross-channel correlation optimization',
                'Memory efficient for long audio sequences'
            ]
        }


class LyCodecEncoder(nn.Module):
    """
    LyCodec encoder with production-grade psychoacoustic modeling.
    
    Transforms 44.1kHz stereo audio into quantized latent representations
    through enhanced psychoacoustic transforms and low-rank fusion layers.
    Uses exclusively Linear Attention for optimal efficiency.
    """
    
    def __init__(self, config: LyCodecConfig):
        super().__init__()
        
        self.config = config
        
        # Stride-32 frontend
        self.frontend = nn.Conv1d(
            in_channels=config.channels,
            out_channels=config.hidden_dim,
            kernel_size=config.frontend_kernel_size,
            stride=config.frontend_stride,
            padding=config.frontend_kernel_size // 2
        )
        
        # Positional encoding for temporal modeling
        frontend_len = math.ceil(config.segment_samples / config.frontend_stride)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, frontend_len, config.hidden_dim) * 0.02
        )
        
        # Transformer layers with cross-level fusion (Linear Attention only)
        self.layers = nn.ModuleList([
            LyCodecTransformerLayer(config, i) for i in range(config.num_layers)
        ])
        
        # Output normalization and projection
        self.output_norm = FastRMSNorm2D(config.hidden_dim)
        self.output_projection = LowRankLinearFusion(
            config.hidden_dim,
            config.hidden_dim,
            rank=config.low_rank_dim
        )
        
        # Production adaptive bit allocator
        self.bit_allocator = ProductionAdaptiveBitAllocator(
            hidden_dim=config.hidden_dim,
            min_bitrate=config.min_bitrate,
            max_bitrate=config.max_bitrate,
            complexity_lookahead=config.complexity_lookahead,
            safety_factor=config.safety_factor
        )
    
    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Encode stereo audio to latent representation using Linear Attention.
        
        Args:
            audio: Input tensor [batch, channels=2, samples=220500]
            
        Returns:
            Tuple of (latent_features, metadata_dict)
        """
        batch_size = audio.shape[0]
        
        # Frontend and positional encoding
        x = self.frontend(audio)  # [batch, hidden_dim, seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, hidden_dim]
        x = x + self.pos_encoding
        
        # Store intermediate features for cross-level fusion
        layer_features = []
        
        # Process through transformer layers (Linear Attention only)
        for i, layer in enumerate(self.layers):
            # Determine cross-level features for fusion layers
            cross_level_features = None
            if i in self.config.fusion_layers and layer_features:
                # Use features from previous fusion layer or early layer
                fusion_idx = max(0, len(layer_features) - 2)
                cross_level_features = layer_features[fusion_idx]
            
            x = layer(x, cross_level_features)
            layer_features.append(x)
        
        # Output normalization and projection
        x = self.output_norm(x)
        latent_features = self.output_projection(x)
        
        # Adaptive bit allocation
        bit_allocation, complexity_metrics = self.bit_allocator(latent_features)
        
        metadata = {
            'bit_allocation': bit_allocation,
            'complexity_metrics': complexity_metrics,
            'layer_features': layer_features[-3:],  # Keep last 3 for analysis
            'original_length': audio.shape[-1]  # Store original audio length
        }
        
        return latent_features, metadata


class LyCodecDecoder(nn.Module):
    """
    LyCodec decoder with DDSP vocoder synthesis.
    
    Reconstructs 44.1kHz stereo audio from quantized latent representations
    through harmonic+noise decomposition and high-performance synthesis.
    Uses exclusively Linear Attention for consistent performance.
    """
    
    def __init__(self, config: LyCodecConfig):
        super().__init__()
        
        self.config = config
        
        # Input projection from quantized features
        self.input_projection = LowRankLinearFusion(
            config.hidden_dim,
            config.hidden_dim,
            rank=config.low_rank_dim
        )
        
        # Transformer layers for latent processing (Linear Attention only)
        self.layers = nn.ModuleList([
            LyCodecTransformerLayer(config, i) for i in range(config.num_layers)
        ])
        
        # DDSP vocoder for high-quality synthesis
        self.vocoder = DDSPVocoder(
            hidden_dim=config.hidden_dim,
            harmonics_count=config.harmonics_count,
            noise_bands=config.noise_bands,
            sample_rate=config.sample_rate,
            channels=config.channels
        )
        
        # Output normalization
        self.output_norm = FastRMSNorm2D(config.hidden_dim)
    
    def forward(self, latent_features: torch.Tensor, 
                metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Decode latent features to stereo audio using Linear Attention.
        
        Args:
            latent_features: Quantized features [batch, seq_len, hidden_dim]
            metadata: Metadata from encoder including bit allocation
            
        Returns:
            Reconstructed audio [batch, channels=2, samples=220500]
        """
        # Input projection
        x = self.input_projection(latent_features)
        
        # Process through transformer layers (Linear Attention only)
        for layer in self.layers:
            x = layer(x)
        
        # Output normalization
        x = self.output_norm(x)
        
        # DDSP vocoder synthesis
        audio = self.vocoder(x, metadata)
        
        # Upsample to original length if needed
        if 'original_length' in metadata:
            original_length = metadata['original_length']
            current_length = audio.shape[-1]
            
            if current_length != original_length:
                # Use interpolation to upsample to original length
                audio = F.interpolate(
                    audio, 
                    size=original_length, 
                    mode='linear', 
                    align_corners=False
                )
        
        return audio


class LyCodecModel(nn.Module):
    """
    Complete LyCodec v2.1 model for production audio compression.
    
    Integrates encoder, continuous bottleneck, and decoder with DDSP vocoder
    for high-quality 44.1kHz stereo audio compression at adaptive bitrates.
    Uses exclusively Linear Attention throughout the architecture.
    """
    
    def __init__(self, config: LyCodecConfig):
        super().__init__()
        
        self.config = config
        
        # Core model components
        self.encoder = LyCodecEncoder(config)
        self.bottleneck_down = BottleneckDown(config.hidden_dim, factor=10)
        self.bottleneck_up = BottleneckUp(config.hidden_dim, factor=10)
        self.decoder = LyCodecDecoder(config)
        
        # Initialize model parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize model parameters with production-grade strategies."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, audio: torch.Tensor, training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for training or inference using Linear Attention.
        
        Args:
            audio: Input stereo audio [batch, channels=2, samples=220500]
            training: Whether in training mode
            
        Returns:
            Dictionary containing reconstructed audio and losses
        """
        # Encode to latent features
        latent_big, metadata = self.encoder(audio)

        # Bottleneck compression and expansion
        latent_small = self.bottleneck_down(latent_big)
        latent_recon = self.bottleneck_up(latent_small)

        # Decode to audio
        reconstructed_audio = self.decoder(latent_recon, metadata)

        return {
            'reconstructed_audio': reconstructed_audio,
            'latent_features': latent_big,
            'bottleneck_latent': latent_small,
            'metadata': metadata
        }
    
    def encode(self, audio: torch.Tensor, bitrate: int = 192) -> bytes:
        """
        Encode audio to compressed bitstream.
        
        Args:
            audio: Input stereo audio [batch, channels=2, samples]
            bitrate: Target bitrate in kbps
            
        Returns:
            Compressed bitstream as bytes
        """
        self.eval()
        with torch.no_grad():
            self.encoder.bit_allocator.set_target_bitrate(bitrate)
            latent_big, metadata = self.encoder(audio)
            latent_small = self.bottleneck_down(latent_big)
            bitstream = self._compress_to_bitstream(latent_small, metadata)
            
        return bitstream
    
    def decode(self, bitstream: bytes) -> torch.Tensor:
        """
        Decode compressed bitstream to audio.
        
        Args:
            bitstream: Compressed audio data
            
        Returns:
            Reconstructed stereo audio [batch, channels=2, samples]
        """
        self.eval()
        with torch.no_grad():
            latent_small, metadata = self._decompress_from_bitstream(bitstream)
            latent_big = self.bottleneck_up(latent_small)
            audio = self.decoder(latent_big, metadata)
            
        return audio
    
    def _compress_to_bitstream(self, features: torch.Tensor, 
                              metadata: Dict[str, torch.Tensor]) -> bytes:
        """Compress quantized features to bitstream (placeholder implementation)."""
        # Production implementation would use entropy coding
        import pickle
        data = {'features': features.cpu(), 'metadata': metadata}
        return pickle.dumps(data)
    
    def _decompress_from_bitstream(self, bitstream: bytes) -> Tuple[torch.Tensor, Dict]:
        """Decompress bitstream to features (placeholder implementation)."""
        # Production implementation would use entropy decoding
        import pickle
        data = pickle.loads(bitstream)
        return data['features'].to(next(self.parameters()).device), data['metadata']
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint with error handling."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            raise
    
    def get_model_size(self) -> int:
        """Calculate total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Estimate memory usage in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        
        return {
            'parameters_mb': param_size / (1024**2),
            'buffers_mb': buffer_size / (1024**2),
            'total_mb': (param_size + buffer_size) / (1024**2)
        }
    
    def get_architecture_info(self) -> Dict[str, Union[str, int, List[str]]]:
        """
        Get comprehensive architecture information emphasizing Linear Attention.
        
        Returns:
            Dictionary with model architecture details
        """
        # Get attention info from a sample transformer layer
        sample_layer_info = self.encoder.layers[0].get_attention_info()
        
        return {
            'model_name': 'LyCodec v2.1',
            'attention_mechanism': 'Linear Attention (Exclusive)',
            'total_parameters': self.get_model_size(),
            'encoder_layers': self.config.num_layers,
            'decoder_layers': self.config.num_layers,
            'attention_heads': self.config.num_attention_heads,
            'hidden_dimension': self.config.hidden_dim,
            'psychoacoustic_bands': self.config.psycho_bands,
            'harmonics_count': self.config.harmonics_count,
            'noise_bands': self.config.noise_bands,
            'audio_specs': {
                'sample_rate': self.config.sample_rate,
                'channels': self.config.channels,
                'segment_length': self.config.segment_length
            },
            'efficiency_features': [
                'Linear Attention scaling O(N×D)',
                'Low-rank matrix factorization',
                'Adaptive bit allocation',
                'Cross-level feature fusion',
                'Orthogonal weight initialization',
                'Memory-efficient bottleneck',
                'DDSP-based synthesis'
            ],
            'attention_advantages': sample_layer_info['advantages'],
            'memory_optimization': 'V100×4 16GB optimized',
            'inference_target': '<8GB memory, RTF 0.55'
        }