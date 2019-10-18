# MelGAN
Unofficial PyTorch implementation of [MelGAN vocoder](https://arxiv.org/abs/1910.06711) (training in progress)

![](./assets/gd.png)

## Prerequisites

Tested on Python 3.6
```bash
pip install -r requirements.txt
```

## Prepare Dataset

- Download dataset for training. This can be any wav files with sample rate 22050Hz. (i.e. LJSpeech was used in paper)
- preprocess: `python preprocess.py -c config/default.yaml -d [data's root path]`
- Edit configuration `yaml` file

## Train & Tensorboard

- `python trainer.py -c [config yaml file] -n [name of the run]`
- `tensorboard --logdir logs/`

## Inference

coming soon

## Results

coming soon


# Implementation Authors

- [Seungwon Park](http://swpark.me) @ MINDsLab (yyyyy@snu.ac.kr, swpark@mindslab.ai)
- Myunchul Joe @ MINDsLab

# License

BSD 3-Clause License.

- [utils/stft.py](./utils/stft.py) by Prem Seetharaman (BSD 3-Clause License)
- [datasets/mel2samp.py](./datasets/mel2samp.py) from https://github.com/NVIDIA/waveglow (BSD 3-Clause License)
- [utils/hparams.py](./utils/hparams.py) from https://github.com/HarryVolek/PyTorch_Speaker_Verification (No License specified)

# Useful resources

- [How to Train a GAN? Tips and tricks to make GANs work](https://github.com/soumith/ganhacks) by Soumith Chintala
- [jaywalnut310/MelGAN-Pytorch](https://github.com/jaywalnut310/MelGAN-Pytorch) by Jaehyeon Kim
