import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad1d(7),
                nn.utils.weight_norm(nn.Conv1d(1, 16, kernel_size=15, stride=1)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.ReflectionPad1d(20),
                nn.utils.weight_norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, groups=4)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.ReflectionPad1d(20),
                nn.utils.weight_norm(nn.Conv1d(64, 256, kernel_size=41, stride=4, groups=16)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.ReflectionPad1d(20),
                nn.utils.weight_norm(nn.Conv1d(256, 1024, kernel_size=41, stride=4, groups=64)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.ReflectionPad1d(20),
                nn.utils.weight_norm(nn.Conv1d(1024, 1024, kernel_size=41, stride=4, groups=256)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.ReflectionPad1d(2),
                nn.utils.weight_norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1)),
                nn.LeakyReLU(0.2),
            ),
            nn.ReflectionPad1d(1),
            nn.utils.weight_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1)),
        ])

    def forward(self, x):
        '''
            returns: (list of 6 features, discriminator score)
            we directly predict score without last sigmoid function
            since we're using Least Squares GAN (https://arxiv.org/abs/1611.04076)
        '''
        features = list()
        for module in self.discriminator:
            x = module(x)
            features.append(x)
        return features[:-1], features[-1]


if __name__ == '__main__':
    model = Discriminator()

    x = torch.randn(3, 1, 22050)
    print(x.shape)

    features, score = model(x)
    for feat in features:
        print(feat.shape)
    print(score.shape)
