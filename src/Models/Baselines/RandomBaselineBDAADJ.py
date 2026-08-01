import torch
from torch import nn

from modeling.Models.Model import TowerModule
from modeling.Models.ModelDatum import ModelOutput, CHANNEL_INPUT, DISPLACEMENT_FIELD, DO_SOFTMAX
from modeling.utils.random_adjustment_utils import estimate_sigma_from_percentiles, generate_x_y_sample


class RandomBaselineBDAADJ(nn.Module):
    """A checkpoint-free BDAADJ baseline that emits a random displacement field.

    Mirrors the BDAADJ alignment models' interface: forward consumes a ModelInput and returns a
    ModelOutput carrying a DISPLACEMENT_FIELD. Each sample gets a single random (dx, dy) drawn
    from the same Rayleigh model used by make_random_adjustment_files, broadcast across the
    field and expressed in field units (pixels / source_dim) so that the downstream
    mask-averaging in TaskModuleBDAADJ.get_building_adjustment recovers a pixel-space offset.
    """
    def __init__(self, initial_x_dim=256, initial_y_dim=256, source_dim=2048):
        nn.Module.__init__(self)
        self.initial_x_dim = initial_x_dim
        self.initial_y_dim = initial_y_dim
        self.source_dim = source_dim
        self.sigma = estimate_sigma_from_percentiles()

    def forward(self, model_input):
        imagery = model_input[CHANNEL_INPUT]
        do_softmax = model_input[DO_SOFTMAX]
        batch_size = imagery.shape[0]

        displacement_field = torch.zeros((batch_size, 2, self.initial_x_dim, self.initial_y_dim))
        for b in range(batch_size):
            dx, dy = generate_x_y_sample(self.sigma)
            displacement_field[b, 0, :, :] = float(dx) / self.source_dim
            displacement_field[b, 1, :, :] = float(dy) / self.source_dim
        displacement_field = displacement_field.to(imagery.device)

        result = ModelOutput()
        result.setField(DISPLACEMENT_FIELD, displacement_field)
        result.setField(DO_SOFTMAX, do_softmax)
        return result


class RandomBaselineBDAADJTowerModule(TowerModule):
    def _load_tower_model(self, hyperparameters, input_channel_map, output_label_map):
        model_parameters = hyperparameters.get("model_parameters", {}) or {}
        return RandomBaselineBDAADJ(
            initial_x_dim=model_parameters.get("initial_x_dim", 256),
            initial_y_dim=model_parameters.get("initial_y_dim", 256),
            source_dim=model_parameters.get("source_dim", 2048),
        )
