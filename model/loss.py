import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def get_loss_g(feat_match, disc_fake, disc_real):
    loss_g = 0.0
    for fake, real in zip(disc_fake, disc_real):
        loss_g += torch.mean(torch.sum(torch.pow(fake[1] - 1.0, 2), dim=[1, 2]))
        for feat_f, feat_r in zip(fake[0], real[0]):
            loss_g += feat_match * torch.mean(torch.abs(feat_f - feat_r))

    return loss_g

@torch.jit.script
def get_loss_d(disc_fake, disc_real):
    loss_d = 0.0
    for fake, real in zip(disc_fake, disc_real):
        loss_d += torch.mean(torch.sum(torch.pow(real[1] - 1.0, 2), dim=[1, 2]))
        loss_d += torch.mean(torch.sum(torch.pow(fake[1], 2), dim=[1, 2]))

    return loss_d
