# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

# Initial Code Sourced from https://github.com/bair-climate-initiative/scale-mae/
# Modified to fit our models

from functools import partial

import timm.models.vision_transformer
import torch
from torch import nn
from timm.models.vision_transformer import PatchEmbed
from modeling.Models.Backbones.ViT.ScaleMAE.util.pos_embed import get_2d_sincos_pos_embed_with_resolution


class PatchEmbedUnSafe(PatchEmbed):
    """Image to Patch Embedding"""

    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.image_size = img_size
        if img_size[0] % patch_size != 0 or img_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = img_size[0] // patch_size, img_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = (patch_size, patch_size)

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        #B, C, H, W = x.shape -- We Dropped this
        # Dropped size check in timm - Original comment from scalemae, we did not do this
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        # We changed Channels to 4 channels instead of 3
        self,
        global_pool=False,
        patch_size=16,
        in_chans=4,
        embed_dim=1024,
        img_size=(1024, 1024),
        out_indicies=None,
        **kwargs
    ):
        super().__init__(embed_dim=embed_dim, **kwargs)

        self.patch_embed = PatchEmbedUnSafe(
            # img_size=kwargs["img_size"], -- We Dropped this
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        ### We Added ###
        self.patch_size = patch_size
        self.d_model = embed_dim
        ### ###

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        # We Added for decoder
        self.out_indices = out_indicies

    def forward_features(self, x, input_res=None, return_featuremaps=False):
        B, _, h, w = x.shape
        x = self.patch_embed(x)
        input_res = input_res.cpu()

        num_patches = int(
            (h * w) / (self.patch_embed.patch_size[0] * self.patch_embed.patch_size[1])
        )
        pos_embed = get_2d_sincos_pos_embed_with_resolution(
            x.shape[-1],
            int(num_patches**0.5),
            input_res,
            cls_token=True,
            device=x.device,
        )

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + pos_embed
        x = self.pos_drop(x)

        # We added featuremaps for decoder
        features = []

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:  # Get intermediate layers
                if i == (len(self.blocks)-1): # We added normalizing for the last layer
                    x = self.norm(x)
                features.append(x[:, 1:]) # remove cls token
        features = [
            y.reshape(B, h // self.patch_size, w // self.patch_size, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
            for y in features
        ]
        ###

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            # outcome = self.fc_norm(x) -- we commented out bc not returning this
        else:
            x = self.norm(x)
            # outcome = x[:, 0] -- we commented out bc not returning this

        # Return Featuremap for decoder
        if return_featuremaps:
            return features
        return x  # Return differnt tensor than originally given

    def forward(self, x, input_res=None, return_featuremaps=False):
        x = self.forward_features(x, input_res=input_res, return_featuremaps=return_featuremaps)
        # We removed #x = self.head(x)
        return x


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
