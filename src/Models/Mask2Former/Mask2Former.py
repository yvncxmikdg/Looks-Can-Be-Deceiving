from modeling.Models.Model import DecoderModule
from modeling.Models.Mask2Former.mask2former_head import Mask2FormerHead


class Mask2Former(DecoderModule):
    def load_decoder_model(self, hyperparameters, output_label_map):
        input_shape = {
            "1": (1024, None, None, 4),
            "2": (1024, None, None, 8),
            "3": (1024, None, None, 16),
            "4": (1024, None, None, 32),
        }
        # The head's width dominates the whole model: at the inherited hidden_dim=2048 /
        # dim_feedforward=4096 (Meta's ViT-7B segmentor config) the head is 674M parameters --
        # more than twice the 303M ViT-L backbone it sits on -- which drives ~8.8GB checkpoints,
        # 40GB of VRAM at batch size 1, and a backward 6x the cost of the forward. Canonical
        # Mask2Former uses 256/2048. Defaults here preserve existing behaviour for any config that
        # does not set them; hidden_dim must stay divisible by 32 (GroupNorm) and by the 16
        # attention heads.
        decoder_parameters = hyperparameters.get("model_parameters", {}).get("decoder_parameters") or {}
        return Mask2FormerHead(
            input_shape=input_shape,
            hidden_dim=decoder_parameters.get("hidden_dim", 2048),
            dim_feedforward=decoder_parameters.get("dim_feedforward", 4096),
            num_classes=len(output_label_map),
            transformer_in_feature=["1", "2", "3", "4"],
        )
