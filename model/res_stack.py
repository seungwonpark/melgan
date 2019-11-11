import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResStack(nn.Module):
    def __init__(self, channel):
        super(ResStack, self).__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(3**i),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=3, dilation=3**i)),
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(1),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=3, dilation=1)),
            )
            for i in range(3)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x

    def remove_weight_norm(self):
        for layer in self.layers:
            nn.utils.remove_weight_norm(layer[1])
            nn.utils.remove_weight_norm(layer[3])
