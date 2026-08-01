from mmseg.registry import MODELS
from mmseg.models import build_segmentor
from modeling.Models.Model import DecoderModule, TowerModule, MaskedTowerModule


class UperNet(DecoderModule, TowerModule):
    def __init__(
        self, hyperparameters=None, input_channel_map=None, output_label_map=None
    ):
        DecoderModule.__init__(self)
        try:
            if "Tower" in hyperparameters["model_type"]:
                TowerModule.__init__(
                    self, hyperparameters, input_channel_map, output_label_map
                )
        except TypeError:
            print("No Tower Module Initialized.")

    def load_decoder_model(self, hyperparameters, output_label_map):
        return MODELS.build(
            dict(
                type="UPerHead",
                in_channels=hyperparameters["model_parameters"]["decoder_parameters"][
                    "in_channels"
                ],
                in_index=tuple(
                    hyperparameters["model_parameters"]["decoder_parameters"][
                        "in_index"
                    ]
                ),
                pool_scales=hyperparameters["model_parameters"]["decoder_parameters"][
                    "pool_scales"
                ],
                channels=hyperparameters["model_parameters"]["decoder_parameters"][
                    "channels"
                ],
                dropout_ratio=hyperparameters["model_parameters"]["decoder_parameters"][
                    "dropout_ratio"
                ],
                num_classes=len(output_label_map),
                norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
                align_corners=hyperparameters["model_parameters"]["decoder_parameters"][
                    "align_corners"
                ],
            )
        )

    # pylint: disable-next=inconsistent-return-statements
    def _load_tower_model(self, hyperparameters, _, output_label_map):
        # pylint: disable-next=no-else-return
        if hyperparameters["model_parameters"]["encoder_name"] == "resnet":
            return build_segmentor(
                dict(
                    type="EncoderDecoder",
                    backbone=dict(
                        type="ResNetV1c",
                        depth=hyperparameters["model_parameters"]["encoder_parameters"][
                            "depth"
                        ],
                        num_stages=hyperparameters["model_parameters"][
                            "encoder_parameters"
                        ]["num_stages"],
                        out_indices=hyperparameters["model_parameters"][
                            "encoder_parameters"
                        ]["out_indices"],
                        dilations=hyperparameters["model_parameters"][
                            "encoder_parameters"
                        ]["dilations"],
                        strides=hyperparameters["model_parameters"][
                            "encoder_parameters"
                        ]["strides"],
                        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
                        norm_eval=hyperparameters["model_parameters"][
                            "encoder_parameters"
                        ]["norm_eval"],
                        style="pytorch",
                        contract_dilation=hyperparameters["model_parameters"][
                            "encoder_parameters"
                        ]["contract_dilation"],
                    ),
                    decode_head=dict(
                        type="UPerHead",
                        in_channels=hyperparameters["model_parameters"][
                            "decoder_parameters"
                        ]["in_channels"],
                        in_index=tuple(
                            hyperparameters["model_parameters"]["decoder_parameters"][
                                "in_index"
                            ]
                        ),
                        pool_scales=hyperparameters["model_parameters"][
                            "decoder_parameters"
                        ]["pool_scales"],
                        channels=hyperparameters["model_parameters"][
                            "decoder_parameters"
                        ]["channels"],
                        dropout_ratio=hyperparameters["model_parameters"][
                            "decoder_parameters"
                        ]["dropout_ratio"],
                        num_classes=len(output_label_map),
                        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
                        align_corners=hyperparameters["model_parameters"][
                            "decoder_parameters"
                        ]["align_corners"],
                    ),
                    auxiliary_head=dict(
                        type="FCNHead",
                        in_channels=hyperparameters["model_parameters"][
                            "auxiliary_head_parameters"
                        ]["in_channels"],
                        in_index=hyperparameters["model_parameters"][
                            "auxiliary_head_parameters"
                        ]["in_index"],
                        channels=hyperparameters["model_parameters"][
                            "auxiliary_head_parameters"
                        ]["channels"],
                        num_convs=hyperparameters["model_parameters"][
                            "auxiliary_head_parameters"
                        ]["num_convs"],
                        concat_input=hyperparameters["model_parameters"][
                            "auxiliary_head_parameters"
                        ]["concat_input"],
                        dropout_ratio=hyperparameters["model_parameters"][
                            "auxiliary_head_parameters"
                        ]["dropout_ratio"],
                        num_classes=len(output_label_map),
                        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
                        align_corners=hyperparameters["model_parameters"][
                            "auxiliary_head_parameters"
                        ]["align_corners"],
                    ),
                )
            )
        elif hyperparameters["model_parameters"]["encoder_name"] == "swin":
            return build_segmentor(
                dict(
                    type="EncoderDecoder",
                    backbone=dict(
                        type="SwinTransformer",
                        pretrain_img_size=224,
                        embed_dims=96,
                        patch_size=4,
                        window_size=7,
                        mlp_ratio=4,
                        depths=[2, 2, 6, 2],
                        num_heads=[3, 6, 12, 24],
                        strides=(4, 2, 2, 2),
                        out_indices=(0, 1, 2, 3),
                        qkv_bias=True,
                        qk_scale=None,
                        patch_norm=True,
                        drop_rate=0.0,
                        attn_drop_rate=0.0,
                        drop_path_rate=0.3,
                        use_abs_pos_embed=False,
                        act_cfg=dict(type="GELU"),
                        norm_cfg=dict(type="LN", requires_grad=True),
                    ),
                    decode_head=dict(
                        type="UPerHead",
                        in_channels=hyperparameters["model_parameters"][
                            "decoder_parameters"
                        ]["in_channels"],
                        in_index=tuple(
                            hyperparameters["model_parameters"]["decoder_parameters"][
                                "in_index"
                            ]
                        ),
                        pool_scales=hyperparameters["model_parameters"][
                            "decoder_parameters"
                        ]["pool_scales"],
                        channels=hyperparameters["model_parameters"][
                            "decoder_parameters"
                        ]["channels"],
                        dropout_ratio=hyperparameters["model_parameters"][
                            "decoder_parameters"
                        ]["dropout_ratio"],
                        num_classes=len(output_label_map),
                        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
                        align_corners=hyperparameters["model_parameters"][
                            "decoder_parameters"
                        ]["align_corners"],
                    ),
                )
            )
        elif hyperparameters["model_parameters"]["encoder_name"] == "convnext":
            return build_segmentor(
                dict(
                    type="EncoderDecoder",
                    backbone=dict(
                        type="mmpretrain.ConvNeXt",
                        arch="tiny",
                        out_indices=(0, 1, 2, 3),
                        drop_path_rate=0.4,
                        layer_scale_init_value=1.0,
                        gap_before_final_norm=False,
                    ),
                    decode_head=dict(
                        type="UPerHead",
                        in_channels=hyperparameters["model_parameters"][
                            "decoder_parameters"
                        ]["in_channels"],
                        in_index=tuple(
                            hyperparameters["model_parameters"]["decoder_parameters"][
                                "in_index"
                            ]
                        ),
                        pool_scales=hyperparameters["model_parameters"][
                            "decoder_parameters"
                        ]["pool_scales"],
                        channels=hyperparameters["model_parameters"][
                            "decoder_parameters"
                        ]["channels"],
                        dropout_ratio=hyperparameters["model_parameters"][
                            "decoder_parameters"
                        ]["dropout_ratio"],
                        num_classes=len(output_label_map),
                        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
                        align_corners=hyperparameters["model_parameters"][
                            "decoder_parameters"
                        ]["align_corners"],
                    ),
                )
            )
        else:
            print("Warning! Backbone for tower module not implemented.")


class MaskedUperNet(UperNet, MaskedTowerModule):
    def __init__(
        self, hyperparameters=None, input_channel_map=None, output_label_map=None
    ):
        UperNet.__init__(self, hyperparameters, input_channel_map, output_label_map)
        try:
            if "Tower" in hyperparameters["model_type"]:
                MaskedTowerModule.__init__(
                    self, hyperparameters, input_channel_map, output_label_map
                )
        except TypeError:
            print("No Tower Module Initialized")

    def _load_tower_model(self, hyperparameters, input_channel_map, output_label_map):
        return UperNet._load_tower_model(
            self, hyperparameters, input_channel_map, output_label_map
        )
