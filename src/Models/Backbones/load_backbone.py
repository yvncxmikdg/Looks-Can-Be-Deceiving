import os
import pathlib
import copy
import torch

from mmseg.registry import MODELS

from modeling.Models.Backbones.ViT.ScaleMAE import models_vit
from modeling.Models.Backbones.ViT.SatMAE.models_vit_temporal import models_dict
from modeling.Models.Backbones.ViT.ConvMAE.convmae import ConvMAE
from modeling.Models.Backbones.ViT.DinoV3.vision_transformer import (
    DinoVisionTransformer,
)
from modeling.Models.Backbones.ViT.DinoV3.dinov3.dinov3_adapter import DINOv3_Adapter
from modeling.Models.Backbones.checkpoint_mod import adapt_checkpoint
from modeling.Models.Backbones.ViT.SatMAE.util.pos_embed import interpolate_pos_embed


def load_checkpoint(backbone_hyperparameters):
    print("Found Pretrained Backbone. Loading Pretrained Backbone...")
    platform = os.name

    if platform == "nt":
        # Load the pretrained ViT
        posix_backup = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath
            checkpoint = torch.load(
            backbone_hyperparameters["model_parameters"][
                "encoder_parameters"
                ]["backbone_path"],
                map_location="cpu",
                )
        finally:
            pathlib.PosixPath = posix_backup
    else:
        checkpoint = torch.load(
        backbone_hyperparameters["model_parameters"]["encoder_parameters"][
            "backbone_path"
            ],
            map_location="cpu",
            )
        
    return checkpoint

def load_scalemae_vit_backbone(backbone_hyperparameters, hyperparmeter_channels_map):
    print("Initializing ViT from ScaleMAE...")
    try:
        output_indicies = backbone_hyperparameters["model_parameters"][
            "decoder_parameters"
            ]["out_indicies"]
    except KeyError:
        print(
            "Did not find out indicies, assuming decoder does not need, passing in empty list..."
        )
        output_indicies = []

    backbone = models_vit.__dict__[
        backbone_hyperparameters["model_parameters"]["encoder_parameters"][
            "vit_model"
        ]
    ](
        in_chans=len(hyperparmeter_channels_map),
        num_classes=backbone_hyperparameters["model_parameters"]["n_cls"],
        drop_path_rate=backbone_hyperparameters["model_parameters"][
            "encoder_parameters"
        ]["drop_path_rate"],
        global_pool=backbone_hyperparameters["model_parameters"][
            "encoder_parameters"
        ]["global_pool"],
        out_indicies=output_indicies,
    )

    if (
        "backbone_path"
        in backbone_hyperparameters["model_parameters"]["encoder_parameters"]
    ):
        checkpoint = load_checkpoint(backbone_hyperparameters)

        checkpoint_model = checkpoint["model"]
        if len(hyperparmeter_channels_map) > 3:
            print(
                 "Input Channels Exceed Pretrained 3, initializing extra weights to zero..."
            )
            checkpoint = adapt_checkpoint(checkpoint)
        backbone.load_state_dict(checkpoint_model, strict=False)
    else:
        print(
            "Did not find Pretrained Backbone, weights will be randomly initalized..."
        )
    print("\tDone")
    return backbone

def load_satmae_vit_backbone(backbone_hyperparameters, hyperparmeter_channels_map):
    print("Initializing temporal ViT from SatMAE...")
    try:
        output_indicies = backbone_hyperparameters["model_parameters"][
            "decoder_parameters"
        ]["out_indicies"]
    except KeyError:
        print(
            "Did not find out indicies, assuming decoder does not need, passing in empty list..."
        )
        output_indicies = []

    backbone = models_dict[backbone_hyperparameters["model_parameters"]["encoder_parameters"]["vit_model"]](
        in_chans=len(hyperparmeter_channels_map),
        img_size=(
                backbone_hyperparameters["training"]["training_parameters"][
                    "tile_x"
                ],
                backbone_hyperparameters["training"]["training_parameters"][
                    "tile_y"
                ],
            ),
        num_classes=1000,#backbone_hyperparameters["model_parameters"]["n_cls"],
        drop_path_rate=backbone_hyperparameters["model_parameters"][
            "encoder_parameters"
        ]["drop_path_rate"],
        global_pool=backbone_hyperparameters["model_parameters"][
            "encoder_parameters"
        ]["global_pool"],
        output_indicies=output_indicies
    )

    if (
        "backbone_path"
        in backbone_hyperparameters["model_parameters"]["encoder_parameters"]
    ):

        checkpoint = load_checkpoint(backbone_hyperparameters)

        checkpoint_model = checkpoint["model"]
        state_dict =backbone.state_dict()

        for k in ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(backbone, checkpoint_model)

        if len(hyperparmeter_channels_map) > 3:
            print(
                "Input Channels Exceed Pretrained 3, initializing extra weights to zero..."
            )
            checkpoint = adapt_checkpoint(checkpoint)
        backbone.load_state_dict(checkpoint_model, strict=False)
    else:
        print(
            "Did not find Pretrained Backbone, weights will be randomly initalized..."
        )
    print("\tDone")
    return backbone

def load_convmae_vit_backbone(backbone_hyperparameters, hyperparmeter_channels_map):
    print("Initializing ConvViT from ConvMAE...")

    backbone = ConvMAE(
        in_chans=len(hyperparmeter_channels_map),
        num_classes=backbone_hyperparameters["model_parameters"]["n_cls"],
        drop_path_rate=backbone_hyperparameters["model_parameters"]["encoder_parameters"]["drop_path_rate"],
        img_size=backbone_hyperparameters["model_parameters"]["encoder_parameters"]["img_input_size"],
        embed_dim=backbone_hyperparameters["model_parameters"]["encoder_parameters"]["embed_dim"],
        patch_size=backbone_hyperparameters["model_parameters"]["encoder_parameters"]["patch_size"],
        mlp_ratio=backbone_hyperparameters["model_parameters"]["encoder_parameters"]["mlp_ratio"],
        num_heads=backbone_hyperparameters["model_parameters"]["encoder_parameters"]["num_heads"],
        depth=backbone_hyperparameters["model_parameters"]["encoder_parameters"]["depth"],
    )

    if (
        "backbone_path"
        in backbone_hyperparameters["model_parameters"]["encoder_parameters"]
    ):

        checkpoint_model = load_checkpoint(backbone_hyperparameters)
            
        state_dict =backbone.state_dict()

        for k in ['pos_embed', 'patch_embed.proj.weight', 'patch_embed3.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(backbone, checkpoint_model)

        if len(hyperparmeter_channels_map) > 3:
            print(
                "Input Channels Exceed Pretrained 3, initializing extra weights to zero..."
            )
            checkpoint = adapt_checkpoint(checkpoint)
        backbone.load_state_dict(checkpoint_model, strict=False)
    else:
        print(
            "Did not find Pretrained Backbone, weights will be randomly initalized..."
        )
    print("\tDone")
    return backbone

def load_dinov3_backbone(backbone_hyperparameters):
    print("Initializing ViT from DinoV3...")

    encoder_parameters = backbone_hyperparameters["model_parameters"]["encoder_parameters"]
    # Honored here rather than through EncoderDecoderModule: that generic path freezes the whole
    # encoder, which for DINOv3 would also freeze the randomly-initialized adapter (SPM,
    # interaction blocks, norms) and leave nothing to train. Defaults to True, so the ViT stays
    # frozen unless a config sets it False.
    freeze_backbone = encoder_parameters.get("freeze_backbone", True)
    print(f"\tDinoV3 ViT backbone will be {'FROZEN' if freeze_backbone else 'FINE-TUNED'}")

    # These arguments must mirror Meta's official dinov3_vitl16 hub entrypoint for the
    # SAT-493M release (dinov3_vitl16_pretrain_sat493m-*.pth). Dropping any of the
    # architecture flags (n_storage_tokens, layerscale_init, mask_k_bias, ...) builds a
    # structurally different network, and a non-strict load would then silently discard
    # the corresponding pretrained tensors (storage tokens, all 48 layerscale gammas).
    dino_backbone = DinoVisionTransformer(
        img_size=backbone_hyperparameters["training"]["training_parameters"][
            "tile_x"
        ],
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        n_storage_tokens=4,
        layerscale_init=1.0e-05,
        mask_k_bias=True,
        norm_layer="layernormbf16",
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        untie_global_and_local_cls_norm=True,
    )
    # The qkv bias_mask buffers are NaN-filled at construction and only defined after
    # init_weights(), so initialize before the (optional) checkpoint load overwrites.
    dino_backbone.init_weights()

    if "backbone_path" in encoder_parameters:
        checkpoint = load_checkpoint(backbone_hyperparameters)

        # The pretrained weights are the whole point of this backbone: fail loudly on
        # any architecture/checkpoint drift instead of training on a partial load.
        dino_backbone.load_state_dict(checkpoint, strict=True)
    else:
        print(
            "Did not find Pretrained Backbone, weights will be randomly initalized..."
        )

    backbone = DINOv3_Adapter(
        backbone=dino_backbone,
        interaction_indexes=[2, 9, 16, 23],
        freeze_backbone=freeze_backbone,
    )
    print("\tDone")
    return backbone

def load_vit_mmseg_backbone(backbone_hyperparameters, hyperparmeter_channels_map):
    print("Initializing VisionTransformer with default config for ViT-L-16...")
    backbone = MODELS.build(
        dict(
            type="VisionTransformer",
            img_size=(
                backbone_hyperparameters["training"]["training_parameters"][
                    "tile_x"
                ],
                backbone_hyperparameters["training"]["training_parameters"][
                    "tile_y"
                ],
            ),
            patch_size=backbone_hyperparameters["model_parameters"][
                "encoder_parameters"
            ]["patch_size"],
            in_channels=len(hyperparmeter_channels_map),
            embed_dims=backbone_hyperparameters["model_parameters"][
                "encoder_parameters"
            ]["embed_dim"],
            num_layers=backbone_hyperparameters["model_parameters"][
                "encoder_parameters"
            ]["n_layers"],
            num_heads=backbone_hyperparameters["model_parameters"][
                "encoder_parameters"
            ]["n_heads"],
            mlp_ratio=backbone_hyperparameters["model_parameters"][
               "encoder_parameters"
            ]["mlp_ratio"],
            out_indices=backbone_hyperparameters["model_parameters"][
                "decoder_parameters"
            ]["out_indicies"],
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            with_cls_token=True,
            norm_cfg=dict(type="LN", eps=1e-6),
            act_cfg=dict(type="GELU"),
            norm_eval=False,
            interpolate_mode="bicubic",
        ),
    )
    print("\tDone")
    return backbone

def load_backbone(
    backbone_name, backbone_hyperparameters, include_mask, input_channel_map
):
    hyperparmeter_channels_map = copy.deepcopy(input_channel_map).dict()
    if not include_mask:
        hyperparmeter_channels_map.pop("mask")

    if backbone_name == "scalemae":
        backbone = load_scalemae_vit_backbone(backbone_hyperparameters, hyperparmeter_channels_map)
    elif backbone_name == "convmae":
        backbone = load_convmae_vit_backbone(backbone_hyperparameters, hyperparmeter_channels_map)
    elif backbone_name == "satmae":
        backbone = load_satmae_vit_backbone(backbone_hyperparameters, hyperparmeter_channels_map)
    elif backbone_name == "dino":
        backbone = load_dinov3_backbone(backbone_hyperparameters)
    elif backbone_name == "vit":
        backbone = load_vit_mmseg_backbone(backbone_hyperparameters, hyperparmeter_channels_map)
    else:
        raise NotImplementedError()

    return backbone
