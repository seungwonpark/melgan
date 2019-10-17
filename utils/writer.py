import random
from torch.utils.tensorboard import SummaryWriter

from .plotting import plot_waveform_to_numpy


class MyWriter(SummaryWriter):
    def __init__(self, logdir, sample_rate):
        super(MyWriter, self).__init__(logdir)
        self.sample_rate = sample_rate
        self.is_first = True

    def log_training(self, g_loss, d_loss, step):
        self.add_scalar('train.g_loss', g_loss, step)
        self.add_scalar('train.d_loss', d_loss, step)

    def log_validation(self, g_loss, d_loss, target, prediction, step):
        self.add_scalar('validation.g_loss', g_loss, step)
        self.add_scalar('validation.d_loss', d_loss, step)

        self.add_audio('raw_audio_predicted', prediction, step, self.sample_rate)
        self.add_image('waveform_predicted', plot_waveform_to_numpy(prediction), step)

        if self.is_first:
            self.add_audio('raw_audio_target', target, step, self.sample_rate)
            self.add_image('waveform_target', plot_waveform_to_numpy(target), step)
            self.is_first = False
