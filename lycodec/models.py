import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .audio import GammatoneFilterbank, psychoacoustic_masking, N_MELS

class RMSNorm(nn.Module):
    """RMS normalization for 2D feature maps"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        
    def forward(self, x):
        var = x.pow(2).mean(dim=1, keepdim=True)
        return x / (var + self.eps).sqrt() * self.weight

class LinearLayer(nn.Module):
    """Low-rank linear transformation for parameter efficiency"""
    def __init__(self, in_features, out_features, rank=None):
        super().__init__()
        if rank is None:
            rank = min(in_features, out_features) // 4
        
        self.rank = rank
        self.U = nn.Linear(in_features, rank, bias=False)
        self.V = nn.Linear(rank, out_features, bias=True)
        
    def forward(self, x):
        return self.V(self.U(x))

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, heads, N, dim_head] each
        
        # Linear Attention: kernel trick with softplus for positivity and stability
        q = torch.clamp(F.softplus(q) * self.scale, min=1e-8)
        k = torch.clamp(F.softplus(k), min=1e-8)
        
        # Compute numerator: q * (k^T * v)
        kv = torch.einsum('b h n d, b h n e -> b h d e', k, v)
        numerator = torch.einsum('b h n d, b h d e -> b h n e', q, kv)
        
        # Compute denominator: q * (k^T * 1)
        k_sum = k.sum(dim=-2)  # [B, heads, dim_head]
        denominator = torch.einsum('b h n d, b h d -> b h n', q, k_sum).unsqueeze(-1) + 1e-8  # Larger eps for stability
        
        # Output with nan check
        out = numerator / denominator
        out = torch.nan_to_num(out)  # Replace nan/inf with 0
        out = out.transpose(1, 2).reshape(B, N, self.heads * self.dim_head)
        return self.to_out(out)

class PsychTransform(nn.Module):
    """
    FIXED: 단순화된 psychoacoustic transform - 모든 파라미터가 항상 사용됨
    """
    def __init__(self, dim, n_filters=64, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.n_filters = n_filters
        
        # Gammatone filterbank - 항상 사용됨
        self.gammatone = GammatoneFilterbank(n_filters=n_filters)
        
        # Attention - 항상 사용됨
        self.attention = Attention(dim, heads=heads)
        
        # 모든 projection이 항상 사용되도록 보장
        self.psych_proj = nn.Linear(n_filters, dim)
        self.norm = RMSNorm(dim)
        
        # 항상 사용되는 learnable parameters
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, log_mel=None):
        """
        FIXED: 모든 파라미터가 항상 사용되는 forward pass
        """
        B, C, H, W = x.shape
        
        # 항상 gammatone을 사용하도록 강제
        if log_mel is not None:
            # Convert log-mel to linear
            mel_spectrum = torch.exp(log_mel)  # [B, n_mels, T]
        else:
            # log_mel이 None이면 x로부터 pseudo mel spectrum 생성
            # x의 spatial dimension을 mel frequency bins로 변환
            x_pooled = F.adaptive_avg_pool2d(x, (N_MELS, W))  # [B, C, N_MELS, W]
            mel_spectrum = torch.mean(x_pooled, dim=1)  # [B, N_MELS, W]
            mel_spectrum = torch.exp(mel_spectrum)  # log-mel처럼 처리
        
        # 항상 gammatone filterbank 적용 - 파라미터 사용 보장
        gammatone_out = self.gammatone(mel_spectrum)  # [B, n_filters, T]
        
        # 항상 psychoacoustic masking 적용
        masking_curve = psychoacoustic_masking(gammatone_out)
        
        # 항상 scale과 bias 적용
        weighted_masking = self.scale * masking_curve + self.bias
        
        # 항상 psych_proj 사용
        psych_weights = self.psych_proj(weighted_masking.transpose(1, 2))  # [B, T, C]
        
        # Interpolate to match feature map size
        psych_weights = F.interpolate(
            psych_weights.transpose(1, 2).unsqueeze(-1), 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(-1)  # [B, C, H, W]
        
        # Apply psychoacoustic weighting - 항상 적용
        x_weighted = x * (1.0 + 0.1 * torch.tanh(psych_weights))
        
        # Reshape for attention - 항상 실행
        x_flat = x_weighted.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        
        # Apply attention - 항상 실행
        attended = self.attention(x_flat)  # [B, H*W, C]
        
        # Reshape back and apply normalization - 항상 실행
        attended = attended.transpose(1, 2).contiguous().view(B, C, H, W)
        return self.norm(attended + x)

class ResBlock(nn.Module):
    """
    FIXED: 단순화된 residual block - 모든 파라미터가 항상 사용됨
    """
    def __init__(self, dim, ff_mult=4, dropout=0.1):
        super().__init__()
        self.psych_attn = PsychTransform(dim)
        self.norm1 = RMSNorm(dim)
        
        # Feedforward network - 항상 사용됨
        hidden_dim = dim * ff_mult
        self.ff = nn.Sequential(
            LinearLayer(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            LinearLayer(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = RMSNorm(dim)
        
    def forward(self, x, log_mel=None):
        """모든 파라미터가 항상 사용되는 forward pass"""
        # Psychoacoustic attention - 항상 실행
        psych_output = self.psych_attn(self.norm1(x), log_mel)
        x = x + psych_output
        
        # Feedforward - 항상 실행
        B, C, H, W = x.shape
        ff_input = x.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [B*H*W, C]
        ff_out = self.ff(ff_input).view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        x = x + self.norm2(ff_out)
        return x

class Encoder(nn.Module):
    """
    FIXED: 단순화된 encoder - 모든 파라미터가 항상 사용됨
    """
    def __init__(self, in_channels=1, base_channels=64, latent_dim=64, n_layers=4):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        # Initial projection - 항상 사용됨
        self.mel_proj = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.phase_proj = nn.Conv2d(in_channels, base_channels // 2, 3, 1, 1)
        self.combined_proj = nn.Conv2d(base_channels + base_channels // 2, base_channels, 1)
        
        # Encoder layers - 모두 항상 사용됨
        self.res_blocks = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        current_dim = base_channels
        
        for i in range(n_layers):
            next_dim = min(current_dim * 2, 512)
            
            self.res_blocks.append(ResBlock(current_dim))
            
            if i < 2:  # Frequency downsampling
                stride = (2, 1)
            else:  # Time downsampling
                stride = (1, 2)
            
            self.conv_layers.append(nn.Conv2d(current_dim, next_dim, 3, stride, 1))
            self.norm_layers.append(RMSNorm(next_dim))
            
            current_dim = next_dim
        
        # Final layers - 항상 사용됨
        self.final_res = ResBlock(current_dim)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 32))
        self.final_conv = nn.Conv2d(current_dim, latent_dim, 1)
        
    def forward(self, log_mel, phase=None, original_log_mel=None):
        """
        FIXED: 모든 파라미터가 항상 사용되는 forward pass
        """
        B, n_mels, T = log_mel.shape
        
        # Convert to 2D feature maps
        log_mel_2d = log_mel.unsqueeze(1)  # [B, 1, n_mels, T] - fixed shape
        
        # 항상 mel_proj 사용
        mel_features = self.mel_proj(log_mel_2d)
        
        # Phase processing - phase가 None이어도 모든 파라미터 사용
        if phase is not None:
            phase_2d = phase.unsqueeze(1)  # [B, 1, n_mels, T]
            phase_features = self.phase_proj(phase_2d)
            combined_features = torch.cat([mel_features, phase_features], dim=1)
            x = self.combined_proj(combined_features)
        else:
            # Phase가 None일 때도 phase_proj와 combined_proj 사용
            dummy_phase = torch.zeros_like(log_mel_2d)
            phase_features = self.phase_proj(dummy_phase)
            combined_features = torch.cat([mel_features, phase_features], dim=1)
            x = self.combined_proj(combined_features)
        
        # 모든 레이어를 항상 통과
        for res_block, conv_layer, norm_layer in zip(
            self.res_blocks, self.conv_layers, self.norm_layers
        ):
            x = res_block(x, original_log_mel)
            x = conv_layer(x)
            x = norm_layer(x)
        
        # Final processing - 항상 실행
        x = self.final_res(x, original_log_mel)
        x = self.adaptive_pool(x)
        x = self.final_conv(x)
        
        return x

class Decoder(nn.Module):
    """
    FIXED: 단순화된 decoder - 모든 파라미터가 항상 사용됨
    """
    def __init__(self, latent_dim=64, base_channels=512, out_channels=N_MELS):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        
        # Initial projection - 항상 사용됨
        self.latent_proj = nn.Conv2d(latent_dim, base_channels, 1)
        
        # Decoder layers - 모두 항상 사용됨
        self.res_blocks = nn.ModuleList()
        self.conv_transpose_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        current_dim = base_channels
        
        for i in range(4):
            next_dim = max(current_dim // 2, 64)
            
            self.res_blocks.append(ResBlock(current_dim))
            
            if i < 2:  # Time upsampling
                scale = 2
                kernel_size = (1, 3)
                stride = (1, scale)
                padding = (0, 1)
                output_padding = (0, 1)
            else:  # Frequency upsampling
                scale = 2
                kernel_size = (3, 1)
                stride = (scale, 1)
                padding = (1, 0)
                output_padding = (1, 0)
            
            self.conv_transpose_layers.append(nn.ConvTranspose2d(
                current_dim, next_dim, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding,
                output_padding=output_padding
            ))
            self.norm_layers.append(RMSNorm(next_dim))
            
            current_dim = next_dim
        
        # Final layers - 항상 사용됨
        self.final_res = ResBlock(current_dim)
        self.mel_head = nn.Conv2d(current_dim, out_channels, 3, 1, 1)
        self.phase_head = nn.Conv2d(current_dim, out_channels, 3, 1, 1)
        
    def forward(self, latent, target_size=None):
        """
        FIXED: 모든 파라미터가 항상 사용되는 forward pass
        """
        x = self.latent_proj(latent)
        
        # 모든 레이어를 항상 통과
        for res_block, conv_layer, norm_layer in zip(
            self.res_blocks, self.conv_transpose_layers, self.norm_layers
        ):
            x = res_block(x)
            x = conv_layer(x)
            x = norm_layer(x)
        
        # Final processing - 항상 실행
        x = self.final_res(x)
        
        # 항상 mel_head와 phase_head 모두 사용
        mel_features = self.mel_head(x)
        phase_features = self.phase_head(x) * math.pi  # Scale to [-π, π]
        
        # Convert back to 1D
        if mel_features.shape[2] == 1:
            log_mel_out = mel_features.squeeze(2)
            phase_out = phase_features.squeeze(2)
        else:
            log_mel_out = F.adaptive_avg_pool2d(mel_features, (1, mel_features.shape[-1])).squeeze(2)
            phase_out = F.adaptive_avg_pool2d(phase_features, (1, phase_features.shape[-1])).squeeze(2)
        
        # Resize to target if specified
        if target_size is not None:
            target_mels, target_t = target_size
            
            if target_t is not None and target_mels > 0 and target_t > 0:
                if log_mel_out.shape[1] != target_mels or log_mel_out.shape[2] != target_t:
                    log_mel_reshaped = log_mel_out.unsqueeze(1)
                    phase_reshaped = phase_out.unsqueeze(1)
                    
                    log_mel_interpolated = torch.nn.functional.interpolate(
                        log_mel_reshaped, size=(target_mels, target_t), 
                        mode='bilinear', align_corners=False
                    )
                    phase_interpolated = torch.nn.functional.interpolate(
                        phase_reshaped, size=(target_mels, target_t), 
                        mode='bilinear', align_corners=False
                    )
                    
                    log_mel_out = log_mel_interpolated.squeeze(1)
                    phase_out = phase_interpolated.squeeze(1)
        
        return log_mel_out, phase_out

class Model(nn.Module):
    """
    FIXED: 단순화된 메인 모델 - 모든 파라미터가 항상 사용됨
    """
    def __init__(self, latent_dim=64, base_channels=64, n_layers=4):
        super().__init__()
        
        self.encoder = Encoder(
            in_channels=1,
            latent_dim=latent_dim, 
            base_channels=base_channels, 
            n_layers=n_layers
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            out_channels=N_MELS
        )
        
    def encode(self, log_mel, phase=None):
        """Encode log-mel and phase features to latent representation"""
        return self.encoder(log_mel, phase, log_mel)
    
    def decode(self, latent, target_size=None):
        """Decode latent to log-mel and phase"""
        return self.decoder(latent, target_size)
    
    def forward(self, log_mel, phase=None):
        """
        FIXED: 모든 파라미터가 항상 사용되는 full encode-decode cycle
        """
        # Encode - 모든 encoder 파라미터 사용
        latent = self.encode(log_mel, phase)
        
        # Decode - 모든 decoder 파라미터 사용
        log_mel_out, phase_out = self.decode(latent, (log_mel.shape[1], log_mel.shape[2]))
        
        return log_mel_out, phase_out, latent

# Backward compatibility
LyCodecModel = Model

# Simplified optimization functions
def apply_tensor_optimizations(model, device='cuda'):
    """Apply basic tensor optimizations"""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return model

def enable_mixed_precision_optimizations():
    """Enable mixed precision optimizations"""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

def apply_torch_compile_optimizations(model, accelerator=None):
    """Apply torch.compile optimizations - disabled for stability"""
    return model

def optimize_interpolation_kernels():
    """Optimize interpolation kernels - placeholder"""
    pass

def profile_training_kernels(model, dummy_input, accelerator=None):
    """Profile training kernels - disabled for stability"""
    pass