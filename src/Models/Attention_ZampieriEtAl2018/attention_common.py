from torch import nn


def init_conv2d_weights(module):
    # Initialize weights for all Conv2d layers with Kaiming (fan-out, ReLU) init.
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
