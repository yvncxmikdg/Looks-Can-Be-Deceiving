from mmseg.registry import MODELS
from modeling.Models.Model import DecoderModule

class Segmenter(DecoderModule):
    def load_decoder_model(self, hyperparameters, output_label_map):
        return MODELS.build(
            dict(
                type="SegmenterMaskTransformerHead",
                in_channels=hyperparameters["model_parameters"]["decoder_parameters"]["in_channels"],
                channels=hyperparameters["model_parameters"]["decoder_parameters"]["channels"],
                num_classes=len(output_label_map),
                num_layers=hyperparameters["model_parameters"]["decoder_parameters"]["num_layers"],
                num_heads=hyperparameters["model_parameters"]["decoder_parameters"]["num_heads"],
                embed_dims=hyperparameters["model_parameters"]["decoder_parameters"]["embed_dims"],
                dropout_ratio=hyperparameters["model_parameters"]["decoder_parameters"]["dropout_ratio"],
            )
        )
