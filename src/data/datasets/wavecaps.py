import os
import torch
import json

from utils.directories import directories, get_dataset_dir
from sacred import Ingredient

wavecaps = Ingredient('wavecaps', ingredients=[directories])

@wavecaps.config
def config():
    folder_name = 'wavecaps'


@wavecaps.capture
def get_wavecaps(folder_name):
    path = os.path.join(get_dataset_dir(), folder_name)
    return WaveCaps(path)


def get_audioset_subset(wave_caps_root):

    with open(os.path.join(wave_caps_root, 'json_files', 'AudioSet_SL', 'as_final.json'), 'r') as f:
        files = json.load(f)['data']

    return [
        {
            'path': os.path.join(wave_caps_root, 'mp3', 'AudioSet_SL', f['id'][:-4] + '.mp3'),
            'caption': f['caption']
        } for f in sorted(files, key=lambda x: x['id'])
    ]


def get_soundbible_subset(wave_caps_root):

    with open(os.path.join(wave_caps_root, 'json_files', 'SoundBible', 'sb_final.json'), 'r') as f:
        files = json.load(f)['data']

    return [
        {
            'path': os.path.join(wave_caps_root, 'mp3', 'SoundBible', f['href'][:-5] + '.mp3'),
            'caption': f['caption']
        } for f in sorted(files, key=lambda x: x['id'])
    ]

def get_bbc_subset(wave_caps_root, filter=False):

    with open(os.path.join(wave_caps_root, 'json_files', 'BBC_Sound_Effects', 'bbc_final.json'), 'r') as f:
        files = json.load(f)['data']

    return [
        {
            'path': os.path.join(wave_caps_root, 'mp3', 'BBC_Sound_Effects', f['id'] + '.mp3'),
            'caption': f['caption']
        } for f in sorted(files, key=lambda x: x['id'])
    ]

def get_freesound_subset(wave_caps_root):

    with open(os.path.join(wave_caps_root, 'json_files', 'FreeSound', 'fsd_final.json'), 'r') as f:
        files = json.load(f)['data']

    return [
        {
            'path': os.path.join(wave_caps_root, 'mp3', 'FreeSound_2', f['id'] + '.mp3'),
            'caption': f['caption']
        } for f in sorted(files, key=lambda x: x['id'])
    ]


class WaveCaps(torch.utils.data.Dataset):

    @wavecaps.capture
    def __init__(self, folder_name):

        root_dir = os.path.join(get_dataset_dir(), folder_name)
        # check parameters
        assert os.path.exists(root_dir), f'Parameter \'audio_caps_root\' is invalid. {root_dir} does not exist.'

        self.wave_caps_root = root_dir

        with open(os.path.join(root_dir, 'missing_files.json'), 'r') as f:
            self.missing = json.load(f)
            self.missing = set(["/".join(m.split('/')[-2:]) for m in self.missing])

        samples_as = get_audioset_subset(self.wave_caps_root)
        samples_soundbible = get_soundbible_subset(self.wave_caps_root)
        samples_fsd = get_freesound_subset(self.wave_caps_root)
        # samples_fsd = get_freesound_2_subset(self.wave_caps_root)
        samples_bbc = get_bbc_subset(self.wave_caps_root)

        # filter

        print("Files per data set:")
        print("AudioSet: ", len(samples_as))
        print("FreeSound: ", len(samples_fsd))
        print("SoundBible: ", len(samples_soundbible))
        print("BBC: ", len(samples_bbc))

        self.samples = samples_as + samples_fsd + samples_bbc + samples_soundbible

        self.samples = [s for s in self.samples if "/".join(s['path'].split('/')[-2:]) not in self.missing]


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        s["keywords"] = ''
        s["idx"] = index
        return s

    def __str__(self):
        return f'{type(self)}'

