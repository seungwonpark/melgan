import os
import glob
import tqdm
import torch
import librosa
import argparse

from model.generator import Generator
from utils.hparams import HParam, load_hparam_str


def main(args):
    checkpoint = torch.load(args.checkpoint_path)
    if args.config is not None:
        hp = HParam(args.config)
    else:
        hp = load_hparam_str(checkpoint['hp_str'])

    model = Generator(hp.audio.n_mel_channels).cuda()
    model.load_state_dict(checkpoint['model_g'])
    model.eval()

    with torch.no_grad():
        for melpath in tqdm.tqdm(glob.glob(os.path.join(args.input_folder, '*.mel'))):
            mel = torch.load(melpath)
            if len(mel.shape) == 2:
                mel = mel.unsqueeze(0)
            mel = mel.cuda()

            audio = model(mel)
            audio = audio.squeeze()
            audio = audio.cpu().detach().numpy()

            out_path = melpath.replace('.mel', '_reconstructed_epoch%04d.wav' % checkpoint['epoch'])
            librosa.output.write_wav(out_path, audio, sr=hp.audio.sampling_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None,
                        help="yaml file for config. will use hp_str from checkpoint if not given.")
    parser.add_argument('-p', '--checkpoint_path', type=str, required=True,
                        help="path of checkpoint pt file for evaluation")
    parser.add_argument('-i', '--input_folder', type=str, required=True,
                        help="directory of mel-spectrograms to invert into raw audio. ")
    args = parser.parse_args()

    main(args)
