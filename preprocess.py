import os
import glob
import tqdm
import torch
import argparse
import numpy as np

from utils.hparams import HParam
from utils.utils import read_wav_np
from utils.audio import MelGen


def main(hp, args):
    melgen = MelGen(hp)

    wav_files = glob.glob(os.path.join(args.data_path, '**', '*.wav'), recursive=True)

    for wavpath in tqdm.tqdm(wav_files, desc='preprocess wav to mel'):
        sr, wav = read_wav_np(wavpath)
        assert sr == hp.audio.sampling_rate, \
            "sample rate mismatch. expected %d, got %d at %s" % \
            (hp.audio.sampling_rate, sr, wavpath)
        
        if len(wav) < hp.audio.segment_length + hp.audio.pad_short:
            wav = np.pad(wav, (0, hp.audio.segment_length + hp.audio.pad_short - len(wav)), \
                    mode='constant', constant_values=0.0)

        mel = melgen.get_normalized_mel(wav)

        mel = torch.from_numpy(mel)

        melpath = wavpath.replace('.wav', '.mel')
        torch.save(mel, melpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for config.")
    parser.add_argument('-d', '--data_path', type=str, required=True,
                        help="root directory of wav files")
    args = parser.parse_args()
    hp = HParam(args.config)

    main(hp, args)
