import random
import subprocess
import numpy as np
from scipy.io.wavfile import read


def get_commit_hash():
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return message.strip().decode('utf-8')

def read_wav_np(path):
    sr, wav = read(path)

    if len(wav.shape) == 2:
        wav = wav[:, 0]

    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0

    wav = wav.astype(np.float32)

    return sr, wav

def cut_wav(L, wav):
    samples = len(wav)
    if samples < L:
        start = random.randint(0, L - samples)
        # if shorter than desired segment length, then let's pad with zero
        wav = np.pad(wav, (start, L - samples - start),
                'constant', constant_values=0.0)
    else:
        start = random.randint(0, samples - L)
        wav = wav[start:start+L]

    #wav /= np.max(np.abs(wav)) # skip normalizing
    return wav
