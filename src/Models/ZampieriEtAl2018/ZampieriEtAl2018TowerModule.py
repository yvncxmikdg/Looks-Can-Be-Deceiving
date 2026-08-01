from modeling.Models.ZampieriEtAl2018.ZampieriEtAl2018 import ZampieriEtAl2018
from modeling.Models.Model import TowerModule


# Tower wrapper that lets the model registry build a bare ZampieriEtAl2018 alignment
# model (it already emits a ModelOutput containing a displacement_field). Used by the
# BDAADJ task, whose model is expected to produce ONLY a displacement field.
class ZampieriEtAl2018TowerModule(TowerModule):
    def _alignment_model_class(self):
        # Subclasses override this to build an alternative alignment model (e.g. attention-augmented).
        return ZampieriEtAl2018

    def _load_tower_model(self, hyperparameters, input_channel_map, output_label_map):
        model_parameters = hyperparameters.get("model_parameters", {}) or {}
        try:
            input_channel_mask_index = input_channel_map.getIdx("mask")
        except KeyError:
            input_channel_mask_index = -1
        return self._alignment_model_class()(
            n_classes=len(output_label_map),
            input_channel_mask_index=input_channel_mask_index,
            output_channel_background_index=output_label_map.getBackgroundClassIdx(),
            chain_steps=model_parameters.get("chain_steps", 4),
            initial_x_dim=model_parameters.get("initial_x_dim", 256),
            initial_y_dim=model_parameters.get("initial_y_dim", 256),
            mask_channel=model_parameters.get("mask_channel", 3),
            chain_combination_strategy=model_parameters.get("chain_combination_strategy", "composition"),
        )
