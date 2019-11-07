# based on https://github.com/keithito/tacotron/blob/master/util/audio.py

import librosa
import numpy as np


class MelGen():
    def __init__(self, hp):
        self.hp = hp

    def get_normalized_mel(self, x):
        x = librosa.feature.melspectrogram(
            y=x,
            sr=self.hp.audio.sampling_rate,
            n_fft=self.hp.audio.filter_length,
            hop_length=self.hp.audio.hop_length,
            win_length=self.hp.audio.win_length,
            n_mels=self.hp.audio.n_mel_channels
        )
        x = self.pre_spec(x)
        return x

    def pre_spec(self, x):
        return self.normalize(librosa.power_to_db(x) - self.hp.audio.ref_level_db)

    def post_spec(self, x):
        return librosa.db_to_power(self.denormalize(x) + self.hp.audio.ref_level_db)

    def normalize(self, x):
        return np.clip(x / -self.hp.audio.min_level_db, -1.0, 0.0) + 1.0

    def denormalize(self, x):
        return (np.clip(x, 0.0, 1.0) - 1.0) * -self.hp.audio.min_level_db
