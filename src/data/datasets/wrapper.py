import torch
import numpy as np

class FixedLengthAudio(torch.utils.data.Dataset):

    def __init__(self, data_set, fixed_length=10, sampling_rate=32000):
        self.data_set = data_set
        self.fixed_length = fixed_length*sampling_rate
        self.sampling_rate = sampling_rate

    def __getitem__(self, item):
        sample = self.data_set[item].copy()
        x = sample.get('audio')
        sample['audio_length'] = min(len(x), self.fixed_length) / self.fixed_length
        if x is None:
            return sample
        if x.shape[-1] < self.fixed_length:
            x = self.__pad__(x, self.fixed_length)
        elif x.shape[-1] > self.fixed_length:
            offset = torch.randint(x.shape[-1] - self.fixed_length + 1, size=(1,)).item()
            x = x[offset:offset+self.fixed_length]
        sample['audio'] = x
        return sample

    def __pad__(self, x, length):
        assert len(x) <= length, 'audio sample is longer than the max length'
        y = np.zeros((self.fixed_length,)).astype(np.float32)
        y[:len(x)] = x
        return y

    def __len__(self):
        return len(self.data_set)
