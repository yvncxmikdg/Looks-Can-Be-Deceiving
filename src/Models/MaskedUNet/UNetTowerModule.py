from modeling.Models.MaskedUNet.UNet import UNet
from modeling.Models.Model import TowerModule, MaskedTowerModule

class UNetTowerModule(TowerModule):
    def _load_tower_model(self, hyperparameters, input_channel_map, output_label_map):
        return UNet(len(input_channel_map), len(output_label_map), hyperparameters=hyperparameters).cpu().to(self._device)

class MaskedUNetTowerModule(MaskedTowerModule):
    def _load_tower_model(self, hyperparameters, input_channel_map, output_label_map):
        return UNet(len(input_channel_map), len(output_label_map), hyperparameters=hyperparameters).cpu().to(self._device)
