import torch
from torch import nn

from modeling.Models.Model import TowerModule, MaskedTowerModule

class RandomBaseline(nn.Module):
    def __init__(self, n_classes, class_weights=None):
        nn.Module.__init__(self)
        self._n_classes = n_classes
        if class_weights is None:
            self.class_weights = torch.tensor([1.0]*n_classes)
        else:
            self.class_weights = torch.tensor(class_weights)

    def forward(self, x):
        return torch.rand(x.shape[0], self._n_classes, x.shape[2], x.shape[3])

class RandomBaselineModel(TowerModule):
    def _load_tower_model(self, hyperparameters, _, output_label_map):

        #Compute the weights
        weights = [1.0]*len(output_label_map)
        for label in output_label_map.getAllLabels():
            weights[output_label_map.getIndex(label)] = hyperparameters["training"]["training_parameters"]["output_class_weights"][label]

        #Initialize the model
        return RandomBaseline(n_classes=len(output_label_map), class_weights=weights)

class MaskedRandomBaselineModel(RandomBaselineModel, MaskedTowerModule):
    def __init__(self, hyperparameters=None, input_channel_map=None, output_label_map=None):
        RandomBaselineModel.__init__(self, hyperparameters, input_channel_map, output_label_map)
        MaskedTowerModule.__init__(self, hyperparameters, input_channel_map, output_label_map)
    def _load_tower_model(self, hyperparameters, input_channel_map, output_label_map):
        return RandomBaselineModel._load_tower_model(self, hyperparameters, input_channel_map, output_label_map)
