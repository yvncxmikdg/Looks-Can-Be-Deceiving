import math
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from modeling.Models.Attention_ZampieriEtAl2018.attention_common import init_conv2d_weights

class CrossAttention(nn.Module):
    """
    Args:
        in_channels: Number of input channels.
        reduction: Reduction factor for attention dimension.
        downsample: Whether to downsample spatial dimensions.
        dropout : Dropout rate for regularization.
        drop_path_rate: Drop path rate for stochastic depth.
        use_positional_encoding: Whether to use positional encoding.
    """

    def __init__(
        self,
        in_channels,
        reduction=16,        # Reduction factor for attention dimension
        downsample=True,     # Whether to downsample spatial dimensions
        dropout=0.1,         # Dropout rate for regularization
        drop_path_rate=0.1,  # Drop path rate for stochastic depth
        use_positional_encoding=False  # Whether to use positional encoding
    ):
        super().__init__()
        self.downsample = downsample
        self.use_positional_encoding = use_positional_encoding
        reduced_channels = in_channels // reduction  # Reduced dimension for Q, K
        value_channels = in_channels // 2           # Reduced dimension for V

        # Normalization layers before projection
        self.norm_q = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.norm_kv = nn.GroupNorm(num_groups=8, num_channels=in_channels)

        # Projection layers for query, key, and value
        self.query = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, value_channels, kernel_size=1)

        # Output projection to restore channel dimension
        self.out_proj = nn.Conv2d(value_channels, in_channels, kernel_size=1)

        # Dropout layers for regularization
        self.attn_dropout = nn.Dropout(dropout)  # Applied to attention weights
        self.output_dropout = nn.Dropout(dropout)  # Applied to output

        # DropPath for stochastic depth
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        # Learnable scaling factor for residual connection
        self.gamma = nn.Parameter(torch.zeros(1))

        # Positional encoding parameters (initialized later if needed)
        self.positional_encoding_q = None
        self.positional_encoding_kv = None

        init_conv2d_weights(self)

    def forward(self, x_q, x_kv):
        """
        Forward pass.

        Args:
            x_q: Query features of shape (B, C, H, W).
            x_kv: Key-value features of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output features after applying cross-attention, with the same
                          shape as x_q (B, C, H, W).
        """

        B, C, H, W = x_q.shape

        # downsample spatial dimensions to reduce computation
        if self.downsample:
            x_q_ds = F.avg_pool2d(x_q, kernel_size=2)  # pylint: disable=not-callable
            x_kv_ds = F.avg_pool2d(x_kv, kernel_size=2)  # pylint: disable=not-callable
        else:
            x_q_ds = x_q
            x_kv_ds = x_kv

        # Add 2D positional encoding if enabled
        if self.use_positional_encoding:
            H_ds, W_ds = x_q_ds.shape[-2:]

            # Create or update positional encoding for query if needed
            if self.positional_encoding_q is None or self.positional_encoding_q.shape != x_q_ds.shape:
                self.positional_encoding_q = nn.Parameter(torch.zeros(1, C, H_ds, W_ds).to(x_q_ds.device))
                nn.init.trunc_normal_(self.positional_encoding_q, std=0.02)
            x_q_ds = x_q_ds + self.positional_encoding_q

            # Create or update positional encoding for key/value if needed
            if self.positional_encoding_kv is None or self.positional_encoding_kv.shape != x_kv_ds.shape:
                self.positional_encoding_kv = nn.Parameter(torch.zeros(1, C, H_ds, W_ds).to(x_kv_ds.device))
                nn.init.trunc_normal_(self.positional_encoding_kv, std=0.02)
            x_kv_ds = x_kv_ds + self.positional_encoding_kv

        # Apply normalization to inputs
        x_q_ds = self.norm_q(x_q_ds)
        x_kv_ds = self.norm_kv(x_kv_ds)

        B, _, H_ds, W_ds = x_q_ds.shape
        N = H_ds * W_ds  # Total number of spatial positions

        # Project to queries, keys, and values
        q = self.query(x_q_ds).view(B, -1, N)  # B x C' x N
        k = self.key(x_kv_ds).view(B, -1, N)   # B x C' x N
        v = self.value(x_kv_ds).view(B, -1, N) # B x C'' x N

        # Scale factor for dot-product attention
        scale = math.sqrt(q.size(1))

        # Compute attention weights: (B x N x C') * (B x C' x N) -> (B x N x N)
        attn = torch.bmm(q.permute(0, 2, 1), k) / scale

        # Apply softmax to get attention distribution
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention weights to values: (B x C'' x N) * (B x N x N) -> (B x C'' x N)
        out = torch.bmm(v, attn.permute(0, 2, 1))

        # Reshape output back to spatial format
        out = out.view(B, -1, H_ds, W_ds)

        # Upsample back to original spatial size
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        # Final projection and dropout
        out = self.out_proj(out)
        out = self.output_dropout(out)

        # Apply residual connection with gamma
        return self.drop_path(self.gamma * out) + x_q
