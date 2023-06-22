import numpy as np
import torch

class RollAudio(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        length = torch.randint(sample['audio'].shape[-1], ()).item()
        sample['audio'] = np.roll(sample['audio'], length, axis=-1)
        return sample

