import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def get_loss_g(feat_match, disc_fake, disc_real):
    loss_g = 0.0
    for (feats_fake, score_fake), (feats_real, _) in zip(disc_fake, disc_real):
        loss_g += torch.mean(torch.sum(torch.pow(score_fake - 1.0, 2), dim=[1, 2]))
        for feat_f, feat_r in zip(feats_fake, feats_real):
            loss_g += feat_match * torch.mean(torch.abs(feat_f - feat_r))

    return loss_g

@torch.jit.script
def get_loss_d(disc_fake, disc_real):
    loss_d = 0.0
    for (_, score_fake), (_, score_real) in zip(disc_fake, disc_real):
        loss_d += torch.mean(torch.sum(torch.pow(score_real - 1.0, 2), dim=[1, 2]))
        loss_d += torch.mean(torch.sum(torch.pow(score_fake, 2), dim=[1, 2]))

    return loss_d
