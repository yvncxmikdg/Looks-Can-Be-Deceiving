import torch

from modeling.Models.Backbones.load_backbone import load_backbone
from modeling.Models.Model import EncoderModule
from modeling.Models.ModelDatum import GSD, CHANNEL_INPUT, TIMESTAMP


class ViTBackboneEncoder(EncoderModule):
    def load_encoder_model(self, hyperparameters, input_channel_map):
        backbone = load_backbone(
            backbone_name=hyperparameters["model_parameters"]["encoder_parameters"][
                "backbone"
            ],
            backbone_hyperparameters=hyperparameters,
            include_mask=hyperparameters["input"]["mask_input"],
            input_channel_map=input_channel_map,
        )
        return backbone

    def prepare_model_input(self, model_input):
        return (model_input[CHANNEL_INPUT],)  # Vanilla ViTs only take in one input


class ScaleMAEBackboneEncoder(ViTBackboneEncoder):
    def prepare_model_input(self, model_input):
        x = model_input[CHANNEL_INPUT]
        gsd_ratio = torch.tensor(model_input[GSD], dtype=torch.float32, device=x.device)
        input_res = torch.ones(len(x), device=x.device).float() * gsd_ratio
        return (x, input_res, True)

class SatMAEBackboneEncoder(ViTBackboneEncoder):
    def prepare_model_input(self, model_input):
        x = model_input[CHANNEL_INPUT]
        timestamp = torch.tensor(model_input[TIMESTAMP], dtype=torch.float32, device=x.device)
        return (x, timestamp)