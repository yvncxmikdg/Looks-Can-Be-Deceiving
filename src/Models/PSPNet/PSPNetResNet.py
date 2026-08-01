from mmseg.models import build_segmentor
from modeling.Models.Model import TowerModule, MaskedTowerModule

class PSPNetResNet(TowerModule):
    # pylint: disable = duplicate-code
    def _load_tower_model(self, hyperparameters, _, output_label_map):
        return build_segmentor(
            dict(
                type="EncoderDecoder",
                backbone=dict(
                    type="ResNetV1c",
                    depth=hyperparameters["model_parameters"]["encoder_parameters"]["depth"],
                    num_stages=hyperparameters["model_parameters"]["encoder_parameters"]["num_stages"],
                    out_indices=hyperparameters["model_parameters"]["encoder_parameters"]["out_indices"],
                    dilations=hyperparameters["model_parameters"]["encoder_parameters"]["dilations"],
                    strides=hyperparameters["model_parameters"]["encoder_parameters"]["strides"],
                    norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                    norm_eval=hyperparameters["model_parameters"]["encoder_parameters"]["norm_eval"],
                    style="pytorch",
                    contract_dilation=hyperparameters["model_parameters"]["encoder_parameters"]["contract_dilation"],
                ),
                decode_head=dict(
                    type="PSPHead",
                    in_channels=hyperparameters["model_parameters"]["decoder_parameters"]["in_channels"],
                    in_index=hyperparameters["model_parameters"]["decoder_parameters"]["in_index"],
                    channels=hyperparameters["model_parameters"]["decoder_parameters"]["channels"],
                    pool_scales=hyperparameters["model_parameters"]["decoder_parameters"]["pool_scales"],
                    dropout_ratio=hyperparameters["model_parameters"]["decoder_parameters"]["dropout_ratio"],
                    num_classes=len(output_label_map),
                    norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                    align_corners=hyperparameters["model_parameters"]["decoder_parameters"]["align_corners"],
                ),
                auxiliary_head=dict(
                    type="FCNHead",
                    in_channels=hyperparameters["model_parameters"]["auxiliary_head_parameters"]["in_channels"],
                    in_index=hyperparameters["model_parameters"]["auxiliary_head_parameters"]["in_index"],
                    channels=hyperparameters["model_parameters"]["auxiliary_head_parameters"]["channels"],
                    num_convs=hyperparameters["model_parameters"]["auxiliary_head_parameters"]["num_convs"],
                    concat_input=hyperparameters["model_parameters"]["auxiliary_head_parameters"]["concat_input"],
                    dropout_ratio=hyperparameters["model_parameters"]["auxiliary_head_parameters"]["dropout_ratio"],
                    num_classes=len(output_label_map),
                    norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                    align_corners=hyperparameters["model_parameters"]["auxiliary_head_parameters"]["align_corners"],
                ),
            )
        )

class MaskedPSPNetResNet(PSPNetResNet, MaskedTowerModule):
    def __init__(self, hyperparameters=None, input_channel_map=None, output_label_map=None):
        PSPNetResNet.__init__(self, hyperparameters, input_channel_map, output_label_map)
        MaskedTowerModule.__init__(self, hyperparameters, input_channel_map, output_label_map)
    def _load_tower_model(self, hyperparameters, input_channel_map, output_label_map):
        return PSPNetResNet._load_tower_model(self, hyperparameters, input_channel_map, output_label_map)
