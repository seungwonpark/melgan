import os
import math
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import traceback

from model.generator import Generator
from model.discriminator import Discriminator
from .utils import get_commit_hash
from .validation import validate


def train(args, pt_dir, chkpt_path, trainset, valset, writer, logger, hp, hp_str):
    model_g = Generator(hp.audio.n_mel_channels).cuda()
    model_d = Discriminator().cuda()

    optim_g = torch.optim.Adam(model_g.parameters(),
        lr=hp.train.adam.lr, betas=(hp.train.adam.beta1, hp.train.adam.beta2))
    optim_d = torch.optim.Adam(model_d.parameters(),
        lr=hp.train.adam.lr, betas=(hp.train.adam.beta1, hp.train.adam.beta2))
    
    githash = get_commit_hash()

    init_epoch = -1
    step = 0

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model_g.load_state_dict(checkpoint['model_g'])
        model_d.load_state_dict(checkpoint['model_d'])
        optim_g.load_state_dict(checkpoint['optim_g'])
        optim_d.load_state_dict(checkpoint['optim_d'])
        step = checkpoint['step']
        init_epoch = checkpoint['epoch']

        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint. Will use new.")

        if githash != checkpoint['githash']:
            logger.warning("Code might be different: git hash is different.")
            logger.warning("%s -> %s" % (checkpoint['githash'], githash))

    else:
        logger.info("Starting new training run.")


    # this accelerates training when the size of minibatch is always consistent.
    # if not consistent, it'll horribly slow down.
    torch.backends.cudnn.benchmark = True

    try:
        model_g.train()
        model_d.train()
        for epoch in itertools.count(init_epoch+1):
            loader = tqdm.tqdm(trainloader, desc='Loading train data')
            for mel, audio in loader:
                mel = mel.cuda()
                audio = audio.cuda()

                # TODO: calculate loss, and backprop
                # TODO: log training

            save_path = os.path.join(pt_dir, '%s_%s_%03d.pt'
                % (args.name, githash, epoch))
            torch.save({
                'model_g': model_g.state_dict(),
                'model_d': model_d.state_dict(),
                'optim_g': optim_g.state_dict(),
                'optim_d': optim_d.state_dict(),
                'step': step,
                'epoch': epoch,
                'hp_str': hp_str,
                'githash': githash,
            }, save_path)
            logger.info("Saved checkpoint to: %s" % save_path)

            # TODO: write validation.py
            # validate(args, model, valset, )

    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
