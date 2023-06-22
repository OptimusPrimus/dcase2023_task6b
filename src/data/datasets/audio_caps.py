import os
import torch
import csv
from utils.directories import directories, get_dataset_dir
from sacred import Ingredient
from data.datasets.audioset import audioset, get_audioset

audiocaps = Ingredient('audiocaps', ingredients=[directories, audioset])

SPLITS = ['train', 'val', 'test']

@audiocaps.config
def config():
    folder_name = 'audiocaps'


@audiocaps.capture
def get_audiocaps(split, folder_name):
    path = os.path.join(get_dataset_dir(), folder_name)
    return AudioCapsDataset(path, split)


class AudioCapsDataset(torch.utils.data.Dataset):

    @audiocaps.capture
    def __init__(self, folder_name, split='train', mp3=False):

        root_dir = os.path.join(get_dataset_dir(), folder_name)
        # check parameters
        assert os.path.exists(root_dir), f'Parameter \'audio_caps_root\' is invalid. {root_dir} does not exist.'
        assert split in SPLITS, f'Parameter \'split\' must be in {SPLITS}.'

        self.audio_caps_root = root_dir
        self.split = split

        if split == 'validation':   # rename validation split
            split = 'val'

        # read ytids and captions from csv
        with open(os.path.join(self.audio_caps_root, 'dataset', f'{split}.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            lines = list(reader)[1:]
        _, self.ytids, _, self.captions = list(map(list, zip(*lines)))
        # sort captions by ytid
        self.ytids, self.captions = list(zip(*sorted(zip(self.ytids, self.captions))))

        # get paths and prediction targets
        self.audioset = get_audioset('train')
        idx = dict(zip(self.audioset.ytids, range(0, len(self.audioset.ytids))))

        self.paths, self.targets = [], []
        for ytid, caption in zip(self.ytids, self.captions):
            i = idx.get('Y' + ytid)
            if i is None:
                continue
            self.paths.append(self.audioset[i]['path'][:-3]+'mp3' if mp3 else self.audioset[i]['path'])
            self.targets.append(caption)
            # self.targets.append(self.audioset[i]['target'])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return {
            'path': self.paths[index],
            # 'ytid': 'Y'+self.ytids[index],
            'target': self.targets[index],
            'caption': self.targets[index],
            'idx': index
        }

    def __str__(self):
        return f'{type(self)}_{self.split}'
