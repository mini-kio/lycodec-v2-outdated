"""
LyCodec v2.5 Continuous Architecture - f10c10 Semantic-Preserving Compression
============================================================================

Complete continuous latent space model with semantic preservation, smooth manifolds,
and f10c10 compression ratio. No discrete tokens - pure continuous representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math
import numpy as np

from .psychoacoustic import PsychoacousticTransform, FastRMSNorm2D
from .vocoder import DDSPVocoder
from .utils import ProductionAdaptiveBitAllocator


@dataclass
class LyCodecV25Config:
    """
    Configuration for LyCodec v2.5 continuous latent space model.
    
    Optimized for f10c10 compression with semantic preservation.
    """
    # Audio specifications
    sample_rate: int = 44100
    channels: int = 2
    segment_length: float = 5.0
    segment_samples: int = 220500  # 44100 * 5.0
    
    # Continuous latent specifications (f10c10)
    compression_ratio: int = 100  # f10c10
    latent_channels: int = 64
    latent_length: int = 2205  # 220500 / 100
    
    # Model architecture dimensions
    hidden_dim: int = 512
    num_layers: int = 8
    num_attention_heads: int = 8
    feedforward_dim: int = 2048
    
    # Psychoacoustic transform configuration
    psycho_bands: int = 64
    psycho_window_size: int = 2048
    psycho_hop_length: int = 512
    
    # Semantic preservation
    semantic_dim: int = 256
    num_semantic_tasks: int = 4
    contrastive_temperature: float = 0.07
    
    # Variational bottleneck
    kl_weight: float = 1e-4
    beta_vae: bool = True
    
    # Information bottleneck
    mi_weight: float = 0.1
    compression_target: float = 0.1
    
    # DDSP vocoder configuration
    harmonics_count: int = 48
    noise_bands: int = 8
    vocoder_hidden_dim: int = 256
    
    # Bit allocation parameters
    min_bitrate: int = 64
    max_bitrate: int = 256
    complexity_lookahead: int = 16
    safety_factor: float = 1.2


class ContinuousBottleneck(nn.Module):
    """
    Continuous bottleneck without quantization for smooth latent space.
    """
    
    def __init__(self, input_dim: int, latent_dim: int, kl_weight: float = 1e-4,
                 semantic_regularization: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.semantic_regularization = semantic_regularization
        
        # Variational parameters
        self.mu_proj = nn.Linear(input_dim, latent_dim)
        self.logvar_proj = nn.Linear(input_dim, latent_dim)
        
        # Semantic feature extraction
        if semantic_regularization:
            self.semantic_proj = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, latent_dim)
            )
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for continuous sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through continuous bottleneck.
        
        Returns:
            latent: Continuous latent representation
            kl_loss: KL divergence loss
            semantic_features: Optional semantic features
        """
        # Compute variational parameters
        mu = self.mu_proj(x)
        logvar = self.logvar_proj(x)
        
        # Sample from latent distribution
        latent = self.reparameterize(mu, logvar)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_loss = torch.mean(kl_loss) * self.kl_weight
        
        # Semantic features
        semantic_features = None
        if self.semantic_regularization:
            semantic_features = self.semantic_proj(x)
        
        return latent, kl_loss, semantic_features


class VariationalBottleneck(nn.Module):
    """
    Variational bottleneck for disentangled continuous representation.
    """
    
    def __init__(self, input_dim: int, latent_dim: int, kl_weight: float = 1e-3,
                 beta_vae: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.beta_vae = beta_vae
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.GELU()
        )
        
        self.mu_layer = nn.Linear(input_dim // 4, latent_dim)
        self.logvar_layer = nn.Linear(input_dim // 4, latent_dim)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to variational parameters."""
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss."""
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return torch.mean(kl) * self.kl_weight


class InformationBottleneck(nn.Module):
    """
    Information bottleneck for optimal compression.
    """
    
    def __init__(self, mi_estimator: str = 'MINE', compression_target: float = 0.1):
        super().__init__()
        self.mi_estimator = mi_estimator
        self.compression_target = compression_target
        
        # MINE network for mutual information estimation
        if mi_estimator == 'MINE':
            self.mine_net = nn.Sequential(
                nn.Linear(128, 64),  # Concatenated features
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
    
    def mine_loss(self, joint: torch.Tensor, marginal: torch.Tensor) -> torch.Tensor:
        """Compute MINE mutual information loss."""
        t_joint = self.mine_net(joint)
        t_marginal = self.mine_net(marginal)
        
        # MINE estimator
        mi_estimate = torch.mean(t_joint) - torch.log(torch.mean(torch.exp(t_marginal)))
        return mi_estimate
    
    def forward(self, latent: torch.Tensor, semantic_features: torch.Tensor) -> torch.Tensor:
        """Compute information bottleneck loss."""
        # Create joint and marginal distributions
        batch_size = latent.shape[0]
        
        # Joint: (latent, semantic) pairs
        joint = torch.cat([
            latent.view(batch_size, -1),
            semantic_features.view(batch_size, -1)
        ], dim=1)
        
        # Marginal: shuffle semantic features
        shuffled_idx = torch.randperm(batch_size)
        marginal = torch.cat([
            latent.view(batch_size, -1),
            semantic_features[shuffled_idx].view(batch_size, -1)
        ], dim=1)
        
        # Compute mutual information
        mi_loss = self.mine_loss(joint, marginal)
        
        # Information bottleneck objective: minimize I(X;Z) - β*I(Y;Z)
        # We want to compress while preserving semantics
        compression_loss = F.relu(mi_loss - self.compression_target)
        
        return compression_loss


class SemanticAwareCompressor(nn.Module):
    """
    Semantic-aware compression that preserves meaning.
    """
    
    def __init__(self, semantic_dim: int, acoustic_dim: int, output_dim: int):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.acoustic_dim = acoustic_dim
        self.output_dim = output_dim
        
        # Separate processing for semantic and acoustic
        self.semantic_processor = nn.Sequential(
            nn.Linear(semantic_dim, semantic_dim // 2),
            nn.GELU(),
            nn.Linear(semantic_dim // 2, output_dim // 2)
        )
        
        self.acoustic_processor = nn.Sequential(
            nn.Linear(acoustic_dim, acoustic_dim // 2),
            nn.GELU(),
            nn.Linear(acoustic_dim // 2, output_dim // 2)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, latent: torch.Tensor, semantic_features: torch.Tensor) -> torch.Tensor:
        """Compress with semantic awareness."""
        # Split latent into semantic and acoustic parts
        semantic_part = semantic_features
        acoustic_part = latent
        
        # Process separately
        semantic_compressed = self.semantic_processor(semantic_part)
        acoustic_compressed = self.acoustic_processor(acoustic_part)
        
        # Fuse
        combined = torch.cat([semantic_compressed, acoustic_compressed], dim=-1)
        output = self.fusion(combined)
        
        return output


class SemanticConsistencyRegularizer(nn.Module):
    """
    Regularizer to maintain semantic consistency across compression stages.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, compressed_semantic: torch.Tensor, 
                original_semantic: torch.Tensor) -> torch.Tensor:
        """Compute semantic consistency loss."""
        # Normalize features
        compressed_norm = F.normalize(compressed_semantic, dim=-1)
        original_norm = F.normalize(original_semantic, dim=-1)
        
        # Cosine similarity
        similarity = torch.sum(compressed_norm * original_norm, dim=-1)
        
        # Encourage high similarity
        consistency_loss = 1.0 - torch.mean(similarity)
        
        return consistency_loss


class AdaptiveContinuousLayer(nn.Module):
    """
    Adaptive continuous compression layer with content-aware processing.
    """
    
    def __init__(self, in_dim: int, out_dim: int, compression_ratio: float = 1.5):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.compression_ratio = compression_ratio
        
        # Adaptive compression network
        self.compressor = nn.Sequential(
            nn.Linear(in_dim, int(in_dim / compression_ratio)),
            nn.GELU(),
            nn.Linear(int(in_dim / compression_ratio), out_dim)
        )
        
        # Content analysis for adaptive processing
        self.content_analyzer = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.GELU(),
            nn.Linear(in_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, semantic_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adaptive compression with content awareness."""
        # Analyze content complexity
        complexity = self.content_analyzer(x)
        
        # Adaptive compression
        compressed = self.compressor(x)
        
        # Modulate based on complexity
        modulated = compressed * (0.5 + 0.5 * complexity)
        
        # Extract semantic features from compressed representation
        layer_semantic = torch.mean(modulated, dim=1, keepdim=True).expand_as(semantic_features)
        
        return modulated, layer_semantic


class SemanticAwareEncoder(nn.Module):
    """
    Semantic-aware encoder with psychoacoustic frontend and multi-task learning.
    """
    
    def __init__(self, config: LyCodecV25Config):
        super().__init__()
        self.config = config
        
        # Psychoacoustic frontend
        self.psycho_frontend = nn.Sequential(
            nn.Conv1d(config.channels, 64, kernel_size=2048, stride=512, padding=1024),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(128, config.hidden_dim, kernel_size=3, stride=1, padding=1)
        )
        
        # Semantic-aware transformer layers
        self.semantic_transformer = nn.ModuleList([
            SemanticTransformerBlock(
                d_model=config.hidden_dim,
                nhead=config.num_attention_heads,
                semantic_dim=config.semantic_dim
            ) for _ in range(6)
        ])
        
        # Continuous bottleneck
        self.continuous_bottleneck = ContinuousBottleneck(
            input_dim=config.hidden_dim,
            latent_dim=config.hidden_dim,
            kl_weight=config.kl_weight,
            semantic_regularization=True
        )
    
    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode audio to semantic-aware latent representation.
        
        Returns:
            latent: Encoded latent features
            semantic_features: Semantic features
            kl_loss: KL divergence loss
        """
        # Psychoacoustic processing
        x = self.psycho_frontend(audio)  # [B, hidden_dim, T]
        x = x.transpose(1, 2)  # [B, T, hidden_dim]
        
        # Semantic-aware encoding
        for transformer in self.semantic_transformer:
            x = transformer(x)
        
        # Continuous bottleneck with semantic regularization
        latent, kl_loss, semantic_features = self.continuous_bottleneck(x)
        
        return latent, semantic_features, kl_loss


class SemanticTransformerBlock(nn.Module):
    """
    Transformer block with semantic awareness.
    """
    
    def __init__(self, d_model: int, nhead: int, semantic_dim: int):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.semantic_dim = semantic_dim
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Semantic branch
        self.semantic_branch = nn.Linear(d_model, semantic_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with semantic processing."""
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feedforward
        ff_out = self.feedforward(x)
        x = self.norm2(x + ff_out)
        
        return x


class HierarchicalContinuousCompressor(nn.Module):
    """
    Multi-stage continuous compression with semantic consistency.
    """
    
    def __init__(self, config: LyCodecV25Config):
        super().__init__()
        self.config = config
        
        # Multi-resolution wavelet transform (simulated)
        self.wavelet_transform = nn.ModuleList([
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=3, stride=2, padding=1)
        ])
        
        # Adaptive compression layers
        self.compression_layers = nn.ModuleList([
            AdaptiveContinuousLayer(
                in_dim=config.hidden_dim, out_dim=384, compression_ratio=1.33
            ),
            AdaptiveContinuousLayer(
                in_dim=384, out_dim=256, compression_ratio=1.5
            )
        ])
        
        # Semantic consistency regularizer
        self.semantic_regularizer = SemanticConsistencyRegularizer()
    
    def forward(self, latent: torch.Tensor, semantic_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hierarchical continuous compression."""
        x = latent.transpose(1, 2)  # [B, hidden_dim, T]
        
        # Multi-scale decomposition
        scales = [x]
        for wavelet_layer in self.wavelet_transform:
            x = wavelet_layer(x)
            scales.append(x)
        
        # Hierarchical compression
        x = scales[-1].transpose(1, 2)  # [B, T, hidden_dim]
        consistency_losses = []
        
        for layer in self.compression_layers:
            x, layer_semantic = layer(x, semantic_features)
            
            # Semantic consistency
            consistency_loss = self.semantic_regularizer(
                layer_semantic, semantic_features
            )
            consistency_losses.append(consistency_loss)
        
        # Final adaptive pooling to target length
        compressed_latent = F.adaptive_avg_pool1d(
            x.transpose(1, 2), self.config.latent_length
        ).transpose(1, 2)  # [B, latent_length, 256]
        
        return compressed_latent, sum(consistency_losses)


class UltraContinuousBottleneck(nn.Module):
    """
    Ultra compression bottleneck for f10c10 achievement.
    """
    
    def __init__(self, config: LyCodecV25Config):
        super().__init__()
        self.config = config
        
        # Variational bottleneck
        self.variational_bottleneck = VariationalBottleneck(
            input_dim=256,
            latent_dim=config.latent_channels,
            kl_weight=config.kl_weight,
            beta_vae=config.beta_vae
        )
        
        # Semantic-aware compression
        self.semantic_compressor = SemanticAwareCompressor(
            semantic_dim=config.semantic_dim,
            acoustic_dim=256,
            output_dim=config.latent_channels
        )
        
        # Information bottleneck
        self.info_bottleneck = InformationBottleneck(
            mi_estimator='MINE',
            compression_target=config.compression_target
        )
    
    def forward(self, compressed_latent: torch.Tensor, 
                semantic_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ultra compression with semantic preservation."""
        # Variational compression
        mu, logvar = self.variational_bottleneck.encode(compressed_latent)
        z = self.variational_bottleneck.reparameterize(mu, logvar)
        
        # Semantic-aware final compression
        ultra_latent = self.semantic_compressor(z, semantic_features)
        
        # Information bottleneck regularization
        ib_loss = self.info_bottleneck(ultra_latent, semantic_features)
        kl_loss = self.variational_bottleneck.kl_loss(mu, logvar)
        
        return ultra_latent, kl_loss + ib_loss


class ContrastiveProjector(nn.Module):
    """
    Contrastive learning projector for semantic similarity.
    """
    
    def __init__(self, input_dim: int, projection_dim: int = 256, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, projection_dim)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project features for contrastive learning."""
        projected = self.projector(features)
        return F.normalize(projected, dim=-1)


class SemanticConsistencyLoss(nn.Module):
    """
    Semantic consistency loss for contrastive learning.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, original_features: torch.Tensor, 
                compressed_features: torch.Tensor) -> torch.Tensor:
        """Compute contrastive semantic consistency loss."""
        # Normalize features
        original_norm = F.normalize(original_features, dim=-1)
        compressed_norm = F.normalize(compressed_features, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(original_norm, compressed_norm.transpose(-2, -1)) / self.temperature
        
        # Labels for contrastive learning (diagonal should be high)
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size, device=logits.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


class SemanticPreservationBranch(nn.Module):
    """
    Multi-task semantic preservation branch.
    """
    
    def __init__(self, config: LyCodecV25Config):
        super().__init__()
        self.config = config
        
        # Multi-task heads
        self.task_heads = nn.ModuleDict({
            'speech_recognition': nn.Linear(config.latent_channels, 1000),
            'music_classification': nn.Linear(config.latent_channels, 100),
            'emotion_recognition': nn.Linear(config.latent_channels, 8),
            'speaker_identification': nn.Linear(config.latent_channels, 1000)
        })
        
        # Contrastive projector
        self.contrastive_projector = ContrastiveProjector(
            input_dim=config.latent_channels,
            projection_dim=256,
            temperature=config.contrastive_temperature
        )
        
        # Semantic consistency loss
        self.semantic_consistency = SemanticConsistencyLoss(
            temperature=config.contrastive_temperature
        )
    
    def forward(self, ultra_latent: torch.Tensor, 
                original_audio: Optional[torch.Tensor] = None) -> Tuple[Dict, torch.Tensor]:
        """Multi-task semantic processing."""
        # Extract semantic outputs
        semantic_outputs = {}
        for task_name, head in self.task_heads.items():
            semantic_outputs[task_name] = head(ultra_latent.mean(dim=1))  # Global average pooling
        
        # Contrastive learning
        contrastive_loss = 0.0
        if original_audio is not None:
            # Extract original semantics (simplified)
            original_semantic = self.extract_original_semantics(original_audio)
            compressed_semantic = self.contrastive_projector(ultra_latent.mean(dim=1))
            
            contrastive_loss = self.semantic_consistency(
                original_semantic, compressed_semantic
            )
        
        return semantic_outputs, contrastive_loss
    
    def extract_original_semantics(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract semantic features from original audio (placeholder)."""
        # This would be replaced with a pre-trained semantic encoder
        return torch.randn(audio.shape[0], 256, device=audio.device)


class SemanticAwareDecoder(nn.Module):
    """
    Semantic-aware decoder with hierarchical expansion.
    """
    
    def __init__(self, config: LyCodecV25Config):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.latent_channels, config.hidden_dim)
        
        # Hierarchical expansion
        self.expansion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim * 2),
                nn.GELU(),
                nn.Linear(config.hidden_dim * 2, config.hidden_dim)
            ),
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim * 2),
                nn.GELU(),
                nn.Linear(config.hidden_dim * 2, config.hidden_dim)
            )
        ])
        
        # DDSP vocoder
        self.ddsp_synthesizer = DDSPVocoder(
            hidden_dim=config.hidden_dim,
            harmonics_count=config.harmonics_count,
            noise_bands=config.noise_bands,
            sample_rate=config.sample_rate,
            channels=config.channels
        )
    
    def forward(self, ultra_latent: torch.Tensor) -> torch.Tensor:
        """Decode continuous latent to audio."""
        # Project to hidden dimension
        x = self.input_projection(ultra_latent)
        
        # Hierarchical expansion
        for expansion_layer in self.expansion_layers:
            x = x + expansion_layer(x)  # Residual connection
        
        # Upsample to target length
        x = F.interpolate(
            x.transpose(1, 2),
            size=self.config.segment_samples // 512,  # Match vocoder input
            mode='linear',
            align_corners=False
        ).transpose(1, 2)
        
        # DDSP synthesis
        audio = self.ddsp_synthesizer(x)
        
        return audio


class LyCodecV25Model(nn.Module):
    """
    Complete LyCodec v2.5 model with continuous latent space and f10c10 compression.
    """
    
    def __init__(self, config: LyCodecV25Config):
        super().__init__()
        self.config = config
        
        # Core components
        self.encoder = SemanticAwareEncoder(config)
        self.hierarchical_compressor = HierarchicalContinuousCompressor(config)
        self.ultra_bottleneck = UltraContinuousBottleneck(config)
        self.semantic_branch = SemanticPreservationBranch(config)
        self.decoder = SemanticAwareDecoder(config)
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize model parameters."""
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
        """Full forward pass for training or inference."""
        # Semantic-aware encoding
        latent, semantic_features, kl_loss = self.encoder(audio)
        
        # Hierarchical compression
        compressed_latent, consistency_loss = self.hierarchical_compressor(
            latent, semantic_features
        )
        
        # Ultra continuous bottleneck
        ultra_latent, bottleneck_loss = self.ultra_bottleneck(
            compressed_latent, semantic_features
        )
        
        # Semantic preservation
        semantic_outputs, contrastive_loss = self.semantic_branch(
            ultra_latent, audio if training else None
        )
        
        # Decode to audio
        reconstructed_audio = self.decoder(ultra_latent)
        
        return {
            'reconstructed_audio': reconstructed_audio,
            'ultra_latent': ultra_latent,
            'semantic_outputs': semantic_outputs,
            'losses': {
                'kl_loss': kl_loss,
                'consistency_loss': consistency_loss,
                'bottleneck_loss': bottleneck_loss,
                'contrastive_loss': contrastive_loss
            }
        }
    
    def encode(self, audio: torch.Tensor, bitrate: int = 128) -> torch.Tensor:
        """Encode audio to continuous latent representation."""
        self.eval()
        with torch.no_grad():
            # Encode through pipeline
            latent, semantic_features, _ = self.encoder(audio)
            compressed_latent, _ = self.hierarchical_compressor(latent, semantic_features)
            ultra_latent, _ = self.ultra_bottleneck(compressed_latent, semantic_features)
            
        return ultra_latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode continuous latent to audio."""
        self.eval()
        with torch.no_grad():
            audio = self.decoder(latent)
        return audio
    
    def semantic_interpolation(self, latent1: torch.Tensor, latent2: torch.Tensor, 
                             alpha: float) -> torch.Tensor:
        """Semantically-aware interpolation in continuous latent space."""
        # Linear interpolation in latent space
        interpolated_latent = alpha * latent1 + (1 - alpha) * latent2
        
        # Optionally apply semantic consistency (simplified)
        return interpolated_latent
    
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
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded LyCodec v2.5 checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            raise