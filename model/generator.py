import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ResStack
#from res_stack import ResStack


class Generator(nn.Module):
    def __init__(self, mel_channel):
        super(Generator, self).__init__()

        self.generator = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(mel_channel, 512, kernel_size=7, stride=1, padding=3)),

            nn.LeakyReLU(),
            nn.utils.weight_norm(nn.ConvTranspose1d(512, 256, kernel_size=16, stride=8, padding=4)),

            ResStack(256),

            nn.LeakyReLU(),
            nn.utils.weight_norm(nn.ConvTranspose1d(256, 128, kernel_size=16, stride=8, padding=4)),

            ResStack(128),

            nn.LeakyReLU(),
            nn.utils.weight_norm(nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)),

            ResStack(64),

            nn.LeakyReLU(),
            nn.utils.weight_norm(nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)),

            ResStack(32),

            nn.LeakyReLU(),
            nn.utils.weight_norm(nn.ConvTranspose1d(32, 1, kernel_size=7, stride=1, padding=3)),
            nn.Tanh(),
        )

    def forward(self, mel):
        return self.generator(mel)


'''
    to run this, fix 
    from . import ResStack
    into
    from res_stack import ResStack
'''
if __name__ == '__main__':
    model = Generator(7)

    x = torch.randn(3, 7, 10)
    print(x.shape)

    y = model(x)
    print(y.shape)
    assert y.shape == torch.Size([3, 1, 2560])
