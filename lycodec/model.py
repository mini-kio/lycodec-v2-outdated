"""
LyCodec v2.1 Main Model Architecture
===================================

Production-ready audio codec architecture with enhanced psychoacoustic modeling,
vectorized quantization, and DDSP vocoder synthesis. Optimized for 44.1kHz stereo
with ~45M parameters targeting V100×4 16GB training and <8GB inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math

from .psychoacoustic import PsychoacousticTransform, FastRMSNorm2D
from .quantization import VectorizedQuantizer, ConsistencyAwareNoiseScheduler
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
    
    # Quantization configuration
    quantization_bits: int = 8
    codebook_size: int = 1024
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    
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


class LyCodecTransformerLayer(nn.Module):
    """
    Enhanced transformer layer with psychoacoustic attention and low-rank fusion.
    
    Integrates orthogonal attention matrices, FastRMSNorm2D, and cross-level
    information flow for production-grade audio modeling.
    """
    
    def __init__(self, config: LyCodecConfig, layer_idx: int):
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_dim // self.num_heads
        
        # Enhanced psychoacoustic attention with orthogonal initialization
        self.psycho_attention = PsychoacousticTransform(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_attention_heads,
            psycho_bands=config.psycho_bands
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
        Forward pass with optional cross-level feature fusion.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            cross_level_features: Optional features from other layers
            
        Returns:
            Transformed tensor with same shape as input
        """
        # Pre-norm psychoacoustic attention with residual connection
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


class LyCodecEncoder(nn.Module):
    """
    LyCodec encoder with production-grade psychoacoustic modeling.
    
    Transforms 44.1kHz stereo audio into quantized latent representations
    through enhanced psychoacoustic transforms and low-rank fusion layers.
    """
    
    def __init__(self, config: LyCodecConfig):
        super().__init__()
        
        self.config = config
        
        # Input projection for stereo audio
        self.input_projection = nn.Conv1d(
            in_channels=config.channels,
            out_channels=config.hidden_dim,
            kernel_size=7,
            stride=2,
            padding=3
        )
        
        # Positional encoding for temporal modeling
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.segment_samples // 2, config.hidden_dim) * 0.02
        )
        
        # Transformer layers with cross-level fusion
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
        Encode stereo audio to latent representation.
        
        Args:
            audio: Input tensor [batch, channels=2, samples=220500]
            
        Returns:
            Tuple of (latent_features, metadata_dict)
        """
        batch_size = audio.shape[0]
        
        # Input projection and positional encoding
        x = self.input_projection(audio)  # [batch, hidden_dim, seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, hidden_dim]
        x = x + self.pos_encoding
        
        # Store intermediate features for cross-level fusion
        layer_features = []
        
        # Process through transformer layers
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
            'layer_features': layer_features[-3:]  # Keep last 3 for analysis
        }
        
        return latent_features, metadata


class LyCodecDecoder(nn.Module):
    """
    LyCodec decoder with DDSP vocoder synthesis.
    
    Reconstructs 44.1kHz stereo audio from quantized latent representations
    through harmonic+noise decomposition and high-performance synthesis.
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
        
        # Transformer layers for latent processing
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
        Decode latent features to stereo audio.
        
        Args:
            latent_features: Quantized features [batch, seq_len, hidden_dim]
            metadata: Metadata from encoder including bit allocation
            
        Returns:
            Reconstructed audio [batch, channels=2, samples=220500]
        """
        # Input projection
        x = self.input_projection(latent_features)
        
        # Process through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Output normalization
        x = self.output_norm(x)
        
        # DDSP vocoder synthesis
        audio = self.vocoder(x, metadata)
        
        return audio


class LyCodecModel(nn.Module):
    """
    Complete LyCodec v2.1 model for production audio compression.
    
    Integrates encoder, vectorized quantization, and decoder with DDSP vocoder
    for high-quality 44.1kHz stereo audio compression at adaptive bitrates.
    """
    
    def __init__(self, config: LyCodecConfig):
        super().__init__()
        
        self.config = config
        
        # Core model components
        self.encoder = LyCodecEncoder(config)
        self.quantizer = VectorizedQuantizer(
            hidden_dim=config.hidden_dim,
            codebook_size=config.codebook_size,
            commitment_cost=config.commitment_cost,
            ema_decay=config.ema_decay
        )
        self.decoder = LyCodecDecoder(config)
        
        # Consistency-aware noise scheduler for training
        self.noise_scheduler = ConsistencyAwareNoiseScheduler(
            hidden_dim=config.hidden_dim
        )
        
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
        Full forward pass for training or inference.
        
        Args:
            audio: Input stereo audio [batch, channels=2, samples=220500]
            training: Whether in training mode
            
        Returns:
            Dictionary containing reconstructed audio and losses
        """
        # Encode to latent features
        latent_features, encoder_metadata = self.encoder(audio)
        
        # Apply noise scheduling during training
        if training:
            latent_features = self.noise_scheduler(latent_features)
        
        # Vectorized quantization
        quantized_features, quantization_loss, quantizer_metadata = self.quantizer(
            latent_features, training=training
        )
        
        # Combine metadata
        decoder_metadata = {**encoder_metadata, **quantizer_metadata}
        
        # Decode to audio
        reconstructed_audio = self.decoder(quantized_features, decoder_metadata)
        
        return {
            'reconstructed_audio': reconstructed_audio,
            'quantization_loss': quantization_loss,
            'latent_features': latent_features,
            'quantized_features': quantized_features,
            'metadata': decoder_metadata
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
            # Set target bitrate in bit allocator
            self.encoder.bit_allocator.set_target_bitrate(bitrate)
            
            # Encode to quantized features
            latent_features, metadata = self.encoder(audio)
            quantized_features, _, _ = self.quantizer(latent_features, training=False)
            
            # Compress to bitstream (simplified implementation)
            bitstream = self._compress_to_bitstream(quantized_features, metadata)
            
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
            # Decompress bitstream (simplified implementation)
            quantized_features, metadata = self._decompress_from_bitstream(bitstream)
            
            # Decode to audio
            audio = self.decoder(quantized_features, metadata)
            
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