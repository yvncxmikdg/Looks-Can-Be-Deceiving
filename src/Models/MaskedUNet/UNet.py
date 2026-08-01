import torch
from torch import nn

from modeling.Models.MaskedUNet.unet_parts import DoubleConv, Down, AttnUp, Up, OutConv

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, hyperparameters):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(in_channels=n_channels,
                              out_channels=hyperparameters["model_parameters"]["layers"]["inc"]["out_channels"],
                              dilation=hyperparameters["model_parameters"]["layers"]["inc"]["dilation"],
                              kernel_size=hyperparameters["model_parameters"]["layers"]["inc"]["kernel_size"],
                              padding_mode=hyperparameters["model_parameters"]["layers"]["inc"]["padding_mode"])

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        last_up_out_channels = None
        for layer in hyperparameters["model_parameters"]["layers"].keys():
            if "down_" in layer:
                l = Down(in_channels=hyperparameters["model_parameters"]["layers"][layer]["in_channels"],
                         out_channels=hyperparameters["model_parameters"]["layers"][layer]["out_channels"],
                         dilation=hyperparameters["model_parameters"]["layers"][layer]["dilation"],
                         kernel_size=hyperparameters["model_parameters"]["layers"][layer]["kernel_size"],
                         padding_mode=hyperparameters["model_parameters"]["layers"][layer]["padding_mode"])
                self.down_layers.append(l)
            if "up_" in layer:
                up_module = AttnUp if hyperparameters["model_parameters"]["layers"][layer]["attention"] else Up
                factor = 2 if hyperparameters["model_parameters"]["layers"][layer]["bilinear"] else 1
                l = up_module(in_channels=hyperparameters["model_parameters"]["layers"][layer]["in_channels"],
                              out_channels=hyperparameters["model_parameters"]["layers"][layer]["out_channels"] // factor,
                              bilinear=hyperparameters["model_parameters"]["layers"][layer]["bilinear"])

                last_up_out_channels = hyperparameters["model_parameters"]["layers"][layer]["out_channels"] // factor
                self.up_layers.append(l)

        #Because this is a unet we assert that there are at least as many down layers as up layers
        assert(len(self.down_layers) == len(self.up_layers))

        self.outc = OutConv(last_up_out_channels, n_classes)

    def _compute_down_convolutions(self, x):
        down_outs = [x]
        for l in self.down_layers:
            down_outs.append(l(down_outs[-1]))
        return down_outs

    def _compute_up_convolutions(self, down_outs):
        intermediate = down_outs[-1]
        for l, d in zip(self.up_layers, down_outs[-2::-1]):
            intermediate = l(intermediate, d)
        return intermediate

    def forward(self, x):
        x_inc = self.inc(x)
        down_outs = self._compute_down_convolutions(x_inc)
        up_convolved_rep = self._compute_up_convolutions(down_outs)
        return self.outc(up_convolved_rep)

    def use_checkpointing(self):
        # pylint: disable=not-callable
        self.inc = torch.utils.checkpoint(self.inc)
        for i, layer in enumerate(self.down_layers):
            self.down_layers[i] = torch.utils.checkpoint(layer)
        for i, layer in enumerate(self.up_layers):
            self.up_layers[i] = torch.utils.checkpoint(layer)

        self.outc = torch.utils.checkpoint(self.outc)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        new_state_dict = {}
        for key, v in state_dict.items():
            k = key.replace("model.", "")

            # For backwards compatiblity
            k = k.replace("down1", "down_layers.0")
            k = k.replace("down2", "down_layers.1")
            k = k.replace("down3", "down_layers.2")
            k = k.replace("down4", "down_layers.3")
            k = k.replace("down5", "down_layers.4")
            k = k.replace("up0", "up_layers.0")
            k = k.replace("up1", "up_layers.1")
            k = k.replace("up2", "up_layers.2")
            k = k.replace("up3", "up_layers.3")
            k = k.replace("up4", "up_layers.4")

            new_state_dict[k] = v

        super().load_state_dict(new_state_dict, strict, assign)
