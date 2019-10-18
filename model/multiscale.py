import torch
import torch.nn as nn
import torch.nn.functional as F

from .discriminator import Discriminator


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList(
            [Discriminator() for _ in range(3)]
        )
        
        self.pooling = nn.ModuleList(
            [nn.Identity()] +
            [nn.AvgPool1d(kernel_size=4, stride=2**i, padding=2) for i in range(1, 3)]
        )

    def forward(self, x):
        ret = list()

        for pool, disc in zip(self.pooling, self.discriminators):
            x = pool(x)
            ret.append(disc(x))

        return ret # [(feat, score), (feat, score), (feat, score)]
