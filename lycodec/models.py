import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .audio import GammatoneFilterbank, psychoacoustic_masking

# IMPROVED: Triton optimization support
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# ========================================================================================
# Triton Optimized Kernels for Model Operations
# ========================================================================================

if HAS_TRITON:
    @triton.jit
    def linear_attention_kernel(
        # Input pointers
        q_ptr, k_ptr, v_ptr,
        # Output pointer
        output_ptr,
        # Dimensions
        batch_size, seq_len, num_heads, head_dim,
        scale,
        # Strides
        q_stride_b, q_stride_s, q_stride_h, q_stride_d,
        k_stride_b, k_stride_s, k_stride_h, k_stride_d,
        v_stride_b, v_stride_s, v_stride_h, v_stride_d,
        out_stride_b, out_stride_s, out_stride_h, out_stride_d,
        # Block sizes
        BLOCK_SIZE_S: tl.constexpr,
        BLOCK_SIZE_D: tl.constexpr,
    ):
        """Triton kernel for optimized linear attention computation"""
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_s = tl.program_id(2)
        
        s_offset = pid_s * BLOCK_SIZE_S + tl.arange(0, BLOCK_SIZE_S)
        s_mask = s_offset < seq_len
        
        d_offset = tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offset < head_dim
        
        # Load Q for current position
        q_offset = (pid_b * q_stride_b + 
                   s_offset[:, None] * q_stride_s + 
                   pid_h * q_stride_h + 
                   d_offset[None, :] * q_stride_d)
        q = tl.load(q_ptr + q_offset, mask=s_mask[:, None] & d_mask[None, :])
        
        # Initialize output accumulator
        output_acc = tl.zeros([BLOCK_SIZE_S, BLOCK_SIZE_D], dtype=tl.float32)
        
        # Compute attention scores and weighted values
        for k_start in range(0, seq_len, BLOCK_SIZE_S):
            k_seq_offset = k_start + tl.arange(0, BLOCK_SIZE_S)
            k_seq_mask = k_seq_offset < seq_len
            
            # Load K and V
            k_offset = (pid_b * k_stride_b + 
                       k_seq_offset[:, None] * k_stride_s + 
                       pid_h * k_stride_h + 
                       d_offset[None, :] * k_stride_d)
            k = tl.load(k_ptr + k_offset, mask=k_seq_mask[:, None] & d_mask[None, :])
            
            v_offset = (pid_b * v_stride_b + 
                       k_seq_offset[:, None] * v_stride_s + 
                       pid_h * v_stride_h + 
                       d_offset[None, :] * v_stride_d)
            v = tl.load(v_ptr + v_offset, mask=k_seq_mask[:, None] & d_mask[None, :])
            
            # Compute attention scores: Q @ K^T
            scores = tl.zeros([BLOCK_SIZE_S, BLOCK_SIZE_S], dtype=tl.float32)
            for d in range(head_dim):
                if d < BLOCK_SIZE_D:
                    q_d = tl.load(q_ptr + (pid_b * q_stride_b + 
                                          s_offset * q_stride_s + 
                                          pid_h * q_stride_h + 
                                          d * q_stride_d), mask=s_mask)
                    k_d = tl.load(k_ptr + (pid_b * k_stride_b + 
                                          k_seq_offset * k_stride_s + 
                                          pid_h * k_stride_h + 
                                          d * k_stride_d), mask=k_seq_mask)
                    scores += q_d[:, None] * k_d[None, :]
            
            scores *= scale
            
            # Apply softmax
            scores_max = tl.max(scores, axis=1, keep_dims=True)
            scores_exp = tl.exp(scores - scores_max)
            scores_sum = tl.sum(scores_exp, axis=1, keep_dims=True)
            attn_weights = scores_exp / (scores_sum + 1e-8)
            
            # Apply attention to values: attn @ V
            for d in range(head_dim):
                if d < BLOCK_SIZE_D:
                    v_d = tl.load(v_ptr + (pid_b * v_stride_b + 
                                          k_seq_offset * v_stride_s + 
                                          pid_h * v_stride_h + 
                                          d * v_stride_d), mask=k_seq_mask)
                    weighted_sum = tl.sum(attn_weights * v_d[None, :], axis=1)
                    output_acc[:, d] += weighted_sum
        
        # Store result
        out_offset = (pid_b * out_stride_b + 
                     s_offset[:, None] * out_stride_s + 
                     pid_h * out_stride_h + 
                     d_offset[None, :] * out_stride_d)
        tl.store(output_ptr + out_offset, output_acc, mask=s_mask[:, None] & d_mask[None, :])

    @triton.jit
    def rms_norm_kernel(
        input_ptr,
        weight_ptr,
        output_ptr,
        batch_size, channels, height, width,
        eps,
        input_stride_b, input_stride_c, input_stride_h, input_stride_w,
        weight_stride,
        output_stride_b, output_stride_c, output_stride_h, output_stride_w,
        BLOCK_SIZE_C: tl.constexpr,
        BLOCK_SIZE_HW: tl.constexpr,
    ):
        """Triton kernel for RMS normalization"""
        pid_b = tl.program_id(0)
        pid_hw = tl.program_id(1)
        
        hw_offset = pid_hw * BLOCK_SIZE_HW + tl.arange(0, BLOCK_SIZE_HW)
        hw_mask = hw_offset < (height * width)
        
        h_offset = hw_offset // width
        w_offset = hw_offset % width
        spatial_mask = (h_offset < height) & (w_offset < width)
        
        # Compute RMS over channel dimension
        mean_square = tl.zeros([BLOCK_SIZE_HW], dtype=tl.float32)
        
        for c_start in range(0, channels, BLOCK_SIZE_C):
            c_offset = c_start + tl.arange(0, BLOCK_SIZE_C)
            c_mask = c_offset < channels
            
            input_offset = (pid_b * input_stride_b + 
                           c_offset[:, None] * input_stride_c + 
                           h_offset[None, :] * input_stride_h + 
                           w_offset[None, :] * input_stride_w)
            
            input_vals = tl.load(input_ptr + input_offset, mask=c_mask[:, None] & spatial_mask[None, :])
            mean_square += tl.sum(input_vals * input_vals, axis=0)
        
        # Compute RMS
        rms = tl.sqrt(mean_square / channels + eps)
        
        # Apply normalization and weight
        for c_start in range(0, channels, BLOCK_SIZE_C):
            c_offset = c_start + tl.arange(0, BLOCK_SIZE_C)
            c_mask = c_offset < channels
            
            input_offset = (pid_b * input_stride_b + 
                           c_offset[:, None] * input_stride_c + 
                           h_offset[None, :] * input_stride_h + 
                           w_offset[None, :] * input_stride_w)
            
            weight_offset = c_offset * weight_stride
            
            input_vals = tl.load(input_ptr + input_offset, mask=c_mask[:, None] & spatial_mask[None, :])
            weight_vals = tl.load(weight_ptr + weight_offset, mask=c_mask)
            
            normalized = input_vals / rms[None, :] * weight_vals[:, None]
            
            output_offset = (pid_b * output_stride_b + 
                           c_offset[:, None] * output_stride_c + 
                           h_offset[None, :] * output_stride_h + 
                           w_offset[None, :] * output_stride_w)
            
            tl.store(output_ptr + output_offset, normalized, mask=c_mask[:, None] & spatial_mask[None, :])

class FastRMSNorm2D(nn.Module):
    """Fast RMS normalization for 2D feature maps with learnable scale - TRITON OPTIMIZED"""
    def __init__(self, dim, eps=1e-6, use_triton=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.use_triton = use_triton and HAS_TRITON
        
        if self.use_triton:
            print(f"FastRMSNorm2D: Using Triton optimization for dim={dim}")
        
    def forward(self, x):
        # x: [B, C, H, W]
        if self.use_triton and x.is_cuda and HAS_TRITON:
            return self._forward_triton(x)
        else:
            return self._forward_pytorch(x)
    
    def _forward_triton(self, x):
        """Triton optimized forward pass"""
        B, C, H, W = x.shape
        output = torch.empty_like(x)
        
        # Define block sizes
        BLOCK_SIZE_C = min(64, C)
        BLOCK_SIZE_HW = min(256, H * W)
        
        # Launch kernel
        grid = (B, triton.cdiv(H * W, BLOCK_SIZE_HW))
        
        rms_norm_kernel[grid](
            x, self.weight, output,
            B, C, H, W, float(self.eps),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            BLOCK_SIZE_C=BLOCK_SIZE_C,
            BLOCK_SIZE_HW=BLOCK_SIZE_HW,
        )
        
        return output
    
    def _forward_pytorch(self, x):
        """PyTorch fallback implementation"""
        # Fixed: Use proper RMS calculation instead of L2-norm
        var = x.pow(2).mean(dim=1, keepdim=True)  # Mean squared over channel dimension only
        return x / (var + self.eps).sqrt() * self.weight

class LowRankLinear(nn.Module):
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

class LinearAttention(nn.Module):
    """Linear attention with Triton optimization for memory efficiency"""
    def __init__(self, dim, heads=8, dim_head=64, use_triton=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.use_triton = use_triton and HAS_TRITON
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        
        # IMPROVED: Check flash attention availability more robustly
        self.use_flash_attention = hasattr(F, 'scaled_dot_product_attention')
        
        if self.use_triton:
            print(f"LinearAttention: Using Triton optimization, heads={heads}, dim_head={dim_head}")
        
    def forward(self, x):
        B, N, C = x.shape
        
        if self.use_triton and x.is_cuda and HAS_TRITON:
            return self._forward_triton(x)
        else:
            return self._forward_pytorch(x)
    
    def _forward_triton(self, x):
        """Triton optimized forward pass"""
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, N, self.heads, self.dim_head).transpose(1, 2), qkv)
        
        # Allocate output
        output = torch.zeros_like(q)
        
        # Define block sizes
        BLOCK_SIZE_S = min(64, N)
        BLOCK_SIZE_D = min(64, self.dim_head)
        
        # Launch kernel
        grid = (B, self.heads, triton.cdiv(N, BLOCK_SIZE_S))
        
        linear_attention_kernel[grid](
            q, k, v, output,
            B, N, self.heads, self.dim_head, float(self.scale),
            q.stride(0), q.stride(2), q.stride(1), q.stride(3),
            k.stride(0), k.stride(2), k.stride(1), k.stride(3),
            v.stride(0), v.stride(2), v.stride(1), v.stride(3),
            output.stride(0), output.stride(2), output.stride(1), output.stride(3),
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )
        
        out = output.transpose(1, 2).contiguous().view(B, N, -1)
        return self.to_out(out)
    
    def _forward_pytorch(self, x):
        """PyTorch fallback implementation"""
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, N, self.heads, -1).transpose(1, 2), qkv)
        
        # IMPROVED: Force consistent computation path for better CPU/GPU consistency
        # Always use scaled_dot_product_attention when available, regardless of device
        if self.use_flash_attention:
            # Use PyTorch 2.1+ scaled_dot_product_attention for consistent results
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False
            )
        else:
            # Standard attention fallback - ensure consistent computation
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            out = torch.matmul(attn_weights, v)
        
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.to_out(out)

class PsychoacousticTransform(nn.Module):
    """
    Psychoacoustic attention module with Triton optimization
    """
    def __init__(self, dim, n_gammatone_filters=64, heads=8, use_triton=True):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.n_filters = n_gammatone_filters
        self.use_triton = use_triton and HAS_TRITON
        
        # Gammatone filterbank for psychoacoustic analysis
        self.gammatone = GammatoneFilterbank(n_filters=n_gammatone_filters, use_triton=use_triton)
        
        # Linear attention with psychoacoustic weighting
        self.attention = LinearAttention(dim, heads=heads, use_triton=use_triton)
        
        # Learnable psychoacoustic weighting with improved activation
        self.psych_proj = nn.Linear(n_gammatone_filters, dim)
        self.norm = FastRMSNorm2D(dim, use_triton=use_triton)
        
        # Learnable scale factor to avoid gradient vanishing
        self.gamma = nn.Parameter(torch.ones(1))
        
        if self.use_triton:
            print(f"PsychoacousticTransform: Using Triton optimization, filters={n_gammatone_filters}")
        
    def forward(self, x, magnitude_spectrum=None):
        """
        Args:
            x: [B, C, H, W] - input features
            magnitude_spectrum: [B, F, T] - magnitude spectrum for psychoacoustic analysis
        """
        B, C, H, W = x.shape
        
        if magnitude_spectrum is not None:
            # Apply gammatone filterbank (automatically uses Triton if available)
            gammatone_out = self.gammatone(magnitude_spectrum)  # [B, n_filters, T]
            
            # Compute psychoacoustic masking (automatically uses Triton if available)
            masking_curve = psychoacoustic_masking(gammatone_out, use_triton=self.use_triton)  # [B, n_filters, T]
            
            # Project to feature dimension and interpolate to match spatial dimensions
            psych_weights = self.psych_proj(masking_curve.transpose(1, 2))  # [B, T, C]
            psych_weights = F.interpolate(
                psych_weights.transpose(1, 2).unsqueeze(-1), 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(-1)  # [B, C, H, W]
            
            # Use tanh with learnable scale instead of sigmoid to avoid saturation
            psych_weights = torch.tanh(psych_weights * self.gamma)
        else:
            psych_weights = torch.ones_like(x)
        
        # Apply psychoacoustic weighting
        x_weighted = x * (1.0 + psych_weights)  # Additive instead of multiplicative
        
        # Reshape for attention
        x_flat = x_weighted.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        
        # Apply linear attention (automatically uses Triton if available)
        attended = self.attention(x_flat)  # [B, H*W, C]
        
        # Reshape back and apply normalization
        attended = attended.transpose(1, 2).view(B, C, H, W)
        return self.norm(attended + x)

class ResidualBlock(nn.Module):
    """Residual block with psychoacoustic attention and low-rank feedforward - TRITON OPTIMIZED"""
    def __init__(self, dim, ff_mult=4, dropout=0.1, use_triton=True):
        super().__init__()
        self.psych_attn = PsychoacousticTransform(dim, use_triton=use_triton)
        self.norm1 = FastRMSNorm2D(dim, use_triton=use_triton)
        
        # Low-rank feedforward
        hidden_dim = dim * ff_mult
        self.ff = nn.Sequential(
            LowRankLinear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            LowRankLinear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = FastRMSNorm2D(dim, use_triton=use_triton)
        
    def forward(self, x, magnitude_spectrum=None):
        # Psychoacoustic attention
        x = x + self.psych_attn(self.norm1(x), magnitude_spectrum)
        
        # Feedforward with channel-wise processing - FIXED contiguous call
        B, C, H, W = x.shape
        ff_input = x.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [B*H*W, C] - Added .contiguous()
        ff_out = self.ff(ff_input).view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        x = x + self.norm2(ff_out)
        return x

class LyEncoder(nn.Module):
    """
    LyCodec Encoder: f10c10 compression (100x total) - TRITON OPTIMIZED
    Time compression: 10x (44.1kHz -> 4.41kHz equivalent)
    Channel compression: 10x (1024 freq bins -> ~102 channels)
    """
    def __init__(self, in_channels=2, base_channels=64, latent_dim=64, n_layers=6, use_triton=True):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.use_triton = use_triton
        
        # Initial projection from complex spectrogram (2 channels: real, imag)
        self.input_proj = nn.Conv2d(in_channels * 2, base_channels, 3, 1, 1)
        
        # Encoder layers with progressive downsampling
        layers = []
        current_dim = base_channels
        
        # f10: Frequency downsampling (1024 -> 102)
        for i in range(3):  # 3 stages: /2, /2, /2.5 ≈ /10
            next_dim = min(current_dim * 2, 512)
            layers.extend([
                ResidualBlock(current_dim, use_triton=use_triton),
                nn.Conv2d(current_dim, next_dim, kernel_size=3, stride=(2, 1), padding=1),
                FastRMSNorm2D(next_dim, use_triton=use_triton)
            ])
            current_dim = next_dim
        
        # c10: Time downsampling (T -> T/10)
        for i in range(2):  # 2 stages: /5, /2 = /10
            next_dim = min(current_dim * 2, 512)
            stride = (1, 5) if i == 0 else (1, 2)
            layers.extend([
                ResidualBlock(current_dim, use_triton=use_triton),
                nn.Conv2d(current_dim, next_dim, kernel_size=3, stride=stride, padding=1),
                FastRMSNorm2D(next_dim, use_triton=use_triton)
            ])
            current_dim = next_dim
        
        # Final compression to latent space
        layers.extend([
            ResidualBlock(current_dim, use_triton=use_triton),
            nn.AdaptiveAvgPool2d((8, 32)),  # Fixed latent spatial size
            nn.Conv2d(current_dim, latent_dim, 1)
        ])
        
        self.layers = nn.ModuleList(layers)
        
        if self.use_triton:
            print(f"LyEncoder: Using Triton optimizations")
        
    def forward(self, complex_spec, magnitude_spectrum=None):
        """
        Args:
            complex_spec: [B, 2, 2, F, T] - complex spectrogram (stereo, real/imag)
            magnitude_spectrum: [B, F, T] - for psychoacoustic analysis
        """
        # Flatten stereo channels: [B, 2, 2, F, T] -> [B, 4, F, T]
        B, C1, C2, F, T = complex_spec.shape
        x = complex_spec.view(B, C1 * C2, F, T)
        
        x = self.input_proj(x)
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ResidualBlock):
                x = layer(x, magnitude_spectrum)
            else:
                x = layer(x)
                
        return x

class LyDecoder(nn.Module):
    """
    LyCodec Decoder: Reconstructs from f10c10 compressed representation - TRITON OPTIMIZED
    """
    def __init__(self, latent_dim=64, base_channels=512, out_channels=2, use_triton=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.use_triton = use_triton
        
        # Initial projection from latent
        self.latent_proj = nn.Conv2d(latent_dim, base_channels, 1)
        
        # Decoder layers with progressive upsampling
        layers = []
        current_dim = base_channels
        
        # c10: Time upsampling (T/10 -> T) - Use transposed convolution for better quality - IMPROVED exact output_padding
        for i in range(2):  # Reverse of encoder: x2, x5
            next_dim = max(current_dim // 2, 64)
            scale = 2 if i == 0 else 5
            kernel_size = (1, scale*2-1)
            padding = (0, scale//2)
            
            # IMPROVED: Calculate exact output_padding using formula
            # F_out = (F_in-1)*stride - 2*pad + kernel_size + output_pad
            # For exact reconstruction: output_pad = stride - 1 - (kernel_size - 1) // 2 + padding 
            output_padding = (0, scale - 1 - (kernel_size[1] - 1) // 2)
            
            layers.extend([
                ResidualBlock(current_dim, use_triton=use_triton),
                nn.ConvTranspose2d(current_dim, next_dim, 
                                 kernel_size=kernel_size, 
                                 stride=(1, scale), 
                                 padding=padding,
                                 output_padding=output_padding),  # IMPROVED: Exact calculation
                FastRMSNorm2D(next_dim, use_triton=use_triton)
            ])
            current_dim = next_dim
        
        # f10: Frequency upsampling (102 -> 1024) - Use transposed convolution
        for i in range(3):  # Reverse: x2.5, x2, x2
            next_dim = max(current_dim // 2, 32)
            layers.extend([
                ResidualBlock(current_dim, use_triton=use_triton),
                nn.ConvTranspose2d(current_dim, next_dim, 
                                 kernel_size=(3, 1), 
                                 stride=(2, 1), 
                                 padding=(1, 0),
                                 output_padding=(1, 0)),
                FastRMSNorm2D(next_dim, use_triton=use_triton)
            ])
            current_dim = next_dim
        
        # Final projection to complex spectrogram with proper activation
        layers.extend([
            ResidualBlock(current_dim, use_triton=use_triton),
            nn.Conv2d(current_dim, out_channels * 2, 3, 1, 1),  # 2 = real, imag (per channel)
            # IMPROVED: Add tanh activation to bound outputs and prevent gradient explosion
            # This helps with phase wrap-around issues and numerical stability
            nn.Tanh()  # Bounded output [-1, 1] for both real and imaginary parts
        ])
        
        self.layers = nn.ModuleList(layers)
        
        if self.use_triton:
            print(f"LyDecoder: Using Triton optimizations")
        
    def forward(self, latent, target_size=None):
        """
        Args:
            latent: [B, latent_dim, H_lat, W_lat] - compressed latent
            target_size: (F, T) - target spectrogram size
        """
        x = self.latent_proj(latent)
        
        for layer in self.layers:
            if isinstance(layer, ResidualBlock):
                x = layer(x)
            else:
                x = layer(x)
        
        # Reshape to complex spectrogram format
        B, C, F, T = x.shape
        # C should be out_channels * 2, reshape to [B, out_channels, 2, F, T]
        real_imag = x.view(B, self.out_channels, 2, F, T)
        
        # Separate real and imaginary parts
        real_part = real_imag[:, :, 0]  # [B, out_channels, F, T]
        imag_part = real_imag[:, :, 1]  # [B, out_channels, F, T]
        
        # Resize to target if specified - IMPROVED: Handle partial target_size
        if target_size is not None:
            target_f, target_t = target_size
            
            # If only frequency is specified, keep original time dimension
            if target_t is None:
                target_t = real_part.shape[-1]
            
            # Ensure we have valid dimensions
            if target_f > 0 and target_t > 0:
                # Use torch.nn.functional explicitly to avoid variable name conflicts
                import torch.nn.functional as TF
                real_part = TF.interpolate(real_part, size=(target_f, target_t), mode='bilinear', align_corners=False)
                imag_part = TF.interpolate(imag_part, size=(target_f, target_t), mode='bilinear', align_corners=False)
        
        return real_part, imag_part

class LyCodecModel(nn.Module):
    """Complete LyCodec model with encoder and decoder - TRITON OPTIMIZED"""
    def __init__(self, latent_dim=64, base_channels=64, n_layers=6, use_triton=True):
        super().__init__()
        self.use_triton = use_triton
        
        self.encoder = LyEncoder(latent_dim=latent_dim, base_channels=base_channels, n_layers=n_layers, use_triton=use_triton)
        self.decoder = LyDecoder(latent_dim=latent_dim, use_triton=use_triton)
        
        if self.use_triton:
            print(f"LyCodecModel: Triton optimizations {'enabled' if HAS_TRITON else 'requested but not available'}")
        
    def encode(self, complex_spec, magnitude_spectrum=None):
        return self.encoder(complex_spec, magnitude_spectrum)
    
    def decode(self, latent, target_size=None):
        return self.decoder(latent, target_size)
    
    def forward(self, complex_spec, magnitude_spectrum=None):
        """Full encode-decode cycle for training"""
        latent = self.encode(complex_spec, magnitude_spectrum)
        real_part, imag_part = self.decode(latent, complex_spec.shape[-2:])  # Use last 2 dims for size
        return real_part, imag_part, latent