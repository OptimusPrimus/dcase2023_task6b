import torch
import pandas as pd
import os

from sacred import Ingredient

from utils.directories import directories, get_dataset_dir

SPLITS = ['development', 'validation', 'evaluation', 'analysis', 'predict']

clotho_v2 = Ingredient('clotho_v2', ingredients=[directories])


@clotho_v2.config
def config():
    # dirs
    folder_name = 'clotho_v2'


@clotho_v2.capture
def get_clotho_v2(split, folder_name):
    root_dir = os.path.join(get_dataset_dir(), folder_name)
    splits = {'train': 'development', 'val': 'validation', 'test': 'evaluation', 'predict': 'predict', 'analysis': 'analysis'}
    assert split in list(splits.keys())
    return Clotho_v2Dataset(root_dir, splits[split])


class Clotho_v2Dataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, split='development'):

        assert os.path.exists(root_dir), f'Parameter \'root_dir\' is invalid. {root_dir} does not exist.'
        assert split in SPLITS, f'Parameter \'split\' must be in {SPLITS}.'
        self.split = split

        self.root_dir = root_dir
        self.files_dir = os.path.join(root_dir, split)
        captions_csv = f'clotho_captions_{split}.csv'
        metadata_csv = f'clotho_metadata_{split}.csv'

        kwargs = {'sep': ';'} if split in ['analysis'] else {}

        if split in ['analysis']:
            files = [file for file in os.listdir(self.files_dir) if file.endswith(".wav")]
            metadata = pd.DataFrame({'file_name': files}, index=files)
            captions = pd.DataFrame({'file_name': files}, index=files)
        else:
            metadata = pd.read_csv(os.path.join(root_dir, metadata_csv), encoding="ISO-8859-1", **kwargs)
            metadata = metadata.set_index('file_name')
            captions = pd.read_csv(os.path.join(root_dir, captions_csv))
            captions = captions.set_index('file_name')

        self.metadata = pd.concat([metadata, captions], axis=1)
        self.metadata.reset_index(inplace=True)
        self.num_captions = 1 if split in ['predict', 'analysis'] else 5

    def __getitem__(self, item):
        sample = dict(self.metadata.iloc[item // self.num_captions].items())
        sample['path'] = os.path.join(self.files_dir, sample['file_name'])
        sample['idx'] = item
        caption_idx = item % self.num_captions
        if f'caption_{caption_idx+1}' in sample:
            sample['caption'] = sample[f'caption_{caption_idx+1}']
            if 'caption_2' in sample:
                del sample['caption_1'], sample['caption_2'], sample['caption_3'], sample['caption_4'], sample['caption_5']
            else:
                del sample['caption_1']
        else:
            sample['caption'] = ''

        if 'sound_id' in sample:
            del sample['sound_id'], sample['sound_link']

        # if 'file_name' in sample:
        #    del sample['file_name']

        if 'start_end_samples' in sample:
            del sample['start_end_samples']

        if 'manufacturer' in sample:
            del sample['manufacturer']

        if 'license' in sample:
            del sample['license']

        return sample

    def __len__(self):
        return len(self.metadata) * self.num_captions

    def __str__(self):
        return f'{type(self)}_{self.split}'

