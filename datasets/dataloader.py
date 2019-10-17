from torch.utils.data import DataLoader

from .mel2samp import Mel2Samp


def create_dataloader(hp, args, train):
    dataset = Mel2Samp(hp.data.train if train else hp.data.validation,
        hp.audio.n_mel_channels, hp.audio.segment_length, hp.audio.filter_length,
        hp.audio.hop_length, hp.audio.win_length, hp.audio.sampling_rate, hp.audio.mel_fmin, hp.audio.mel_fmax)

    if train:
        return DataLoader(dataset=dataset, batch_size=hp.train.batch_size, shuffle=True,
            num_workers=hp.train.num_workers, pin_memory=True, drop_last=True)
    else:
        return DataLoader(dataset=dataset, batch_size=hp.train.batch_size, shuffle=False,
            num_workers=hp.train.num_workers, pin_memory=True, drop_last=True)
