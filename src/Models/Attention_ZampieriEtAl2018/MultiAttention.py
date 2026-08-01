import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from modeling.Models.Attention_ZampieriEtAl2018.attention_common import init_conv2d_weights

class MultiHeadSelfAttention2D(nn.Module):
    def __init__(self, in_channels, num_heads=4, dropout=0.1, drop_path_rate=0.1, use_positional_encoding=False):
        """
        Initialize the 2D Multi-Head Self-Attention module.

        Args:
            in_channels: Number of input channels (C)
            num_heads: Number of attention heads
            dropout: Dropout rate for attention weights and outputs
            drop_path_rate: Stochastic depth rate
            use_positional_encoding: Whether to use positional encoding
        """
        super().__init__()
        # Ensure the number of input channels is divisible by number of heads
        assert in_channels % num_heads == 0, f"in_channels ({in_channels}) must be divisible by num_heads ({num_heads})"

        # Store parameters
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.head_dim = in_channels // num_heads  # Dimension of each attention head
        self.scale = self.head_dim ** -0.5  # Scaling factor for attention scores
        self.use_positional_encoding = use_positional_encoding

        # 1x1 convolution for projecting input to query, key, value representations
        # Output channels = in_channels * 3 (for Q, K, V)
        self.qkv_proj = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)

        # 1x1 convolution for projecting attention output back to original dimensions
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Dropout layers for regularization
        self.dropout = nn.Dropout(dropout)

        # Stochastic depth dropout (DropPath) or identity if not used
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        # Group normalization for input features
        self.norm = nn.GroupNorm(num_groups=8, num_channels=in_channels)

        # Positional encoding initialized as None (will be created if needed)
        self.positional_encoding = None

        # Initialize weights
        init_conv2d_weights(self)

    def forward(self, x):
        """
        Forward pass of the 2D Multi-Head Self-Attention module.

        Args:
            x: Input tensor of shape (B, C, H, W)
                B = Batch size
                C = Channels
                H = Height
                W = Width

        Returns:
            Output tensor of same shape as input
        """
        B, C, H, W = x.shape
        N = H * W  # Total number of spatial positions

        # Store input for residual connection
        residual = x

        # Add positional encoding if enabled
        if self.use_positional_encoding:
            # Create positional encoding if it doesn't exist or has wrong shape
            if self.positional_encoding is None or self.positional_encoding.shape != x.shape:
                # Initialize positional encoding as learnable parameter
                self.positional_encoding = nn.Parameter(torch.zeros(1, C, H, W).to(x.device))
                # Initialize with truncated normal distribution
                nn.init.trunc_normal_(self.positional_encoding, std=0.02)

            # Add positional encoding to input
            x = x + self.positional_encoding

        # Apply normalization to input
        x_norm = self.norm(x)

        # Project input to query, key, value representations using 1x1 convolution
        qkv = self.qkv_proj(x_norm)

        # Split concatenated channels into query, key, value
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # Reshape and permute for multi-head attention computation
        # Original shape: (B, C, H, W) -> (B, C, N) where N = H*W
        # After reshape and permute:
        # q: (B, heads, N, head_dim) - Query
        # k: (B, heads, head_dim, N) - Key (transposed for matrix multiplication)
        # v: (B, heads, N, head_dim) - Value
        q = q.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, self.head_dim, N)
        v = v.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        # Compute attention scores: (B, heads, N, N)
        # Attention formula: softmax(Q * K^T / sqrt(d_k))
        attn = torch.matmul(q, k) * self.scale  # Scale to prevent softmax saturation
        attn = F.softmax(attn, dim=-1)  # Apply softmax across last dimension
        attn = self.dropout(attn)  # Apply dropout for regularization

        # Apply attention weights to values
        out = torch.matmul(attn, v)  # (B, heads, N, head_dim)

        # Reshape output back to original format
        # (B, heads, N, head_dim) -> (B, C, H, W)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)

        # Project back to original channel dimension
        out = self.out_proj(out)

        # Apply dropout
        out = self.dropout(out)

        # Add residual connection and apply drop path
        return self.drop_path(out) + residual
