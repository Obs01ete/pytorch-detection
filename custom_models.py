#

import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SimpleBackbone(nn.Module):

    def __init__(self):
        super().__init__()

        base_multiplier = 32

        self.config = (
            (1.0, 7, 2, False),

            (2.0, 3, 2, False),
            (1.0, 1, 1, False),
            (2.0, 3, 1, True),  # <- branch

            (2.0, 3, 2, False),
            (1.0, 1, 1, False),
            (2.0, 3, 1, True),  # <- branch

            (2.0, 3, 2, False),
            (1.0, 1, 1, False),
            (2.0, 3, 1, True),  # <- branch

            (2.0, 3, 2, False),
            (1.0, 1, 1, False),
            (2.0, 3, 1, True),  # <- branch
        )

        in_planes = 3
        self.layers = nn.ModuleList()
        for ch_mul, kernel_size, stride, is_branch in self.config:
            out_planes = int(base_multiplier * ch_mul)
            conv = ConvBlock(in_planes, out_planes, kernel_size=kernel_size, stride=stride)
            self.layers.append(conv)
            in_planes = out_planes

    def forward(self, x):

        branches = []
        for conv, is_branch in zip(self.layers, [c[3] for c in self.config]):
            x = conv(x)
            if is_branch:
                branches.append(x)

        return branches


def simple_backbone(**kwargs):
    """Constructs a simple backbone."""
    model = SimpleBackbone(**kwargs)
    return model
