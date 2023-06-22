import os
import numpy as np
import torch
import csv
import tqdm
import json

from sacred import Ingredient

from utils.directories import directories, get_dataset_dir

audioset = Ingredient('audioset', ingredients=[directories])

SPLITS = ['train', 'evaluation', 'balanced']
N_CLASSES = 527

cached_audiosets = None

@audioset.config
def config():
    # dirs
    folder_name = 'audioset'
    # data set configuration
    update_parent_labels = False


@audioset.capture
def get_audioset(split, folder_name, update_parent_labels, dev=False):
    root_dir = os.path.join(get_dataset_dir(), folder_name)
    if dev and split == 'train':
        # avoid loading training set as it is quite large
        split = 'balanced'

    global cached_audiosets

    if cached_audiosets is None:
        cached_audiosets = {
            'train': AudioSetDataset(root_dir, split='train', update_parent_labels=update_parent_labels),
            'balanced': AudioSetDataset(root_dir, split='balanced', update_parent_labels=update_parent_labels),
            'evaluation': AudioSetDataset(root_dir, split='evaluation', update_parent_labels=update_parent_labels),
        }

    return cached_audiosets[split]


@audioset.capture
def get_audiosets(dev=False):
    train = get_audioset('train', dev=dev)
    balanced = get_audioset('balanced', dev=dev)
    evaluation = get_audioset('evaluation', dev=dev)
    return train, balanced, evaluation


@audioset.capture
def get_sample_weight_estimates(dataset):
    labels = np.stack([s['target'] for s in dataset])
    frequencies = 1000. / (labels.sum(axis=0, keepdims=True) + 100)
    weights = (labels * frequencies).sum(axis=1)
    return weights


@audioset.capture
def get_class_labels_ids(folder_name):
    root_dir = os.path.join(get_dataset_dir(), folder_name)
    assert os.path.exists(root_dir), f'Parameter \'root_dir\' is invalid. {root_dir} does not exist.'

    metadata_dir = os.path.join(root_dir, 'metadata')

    with open(os.path.join(metadata_dir, 'class_labels_indices.csv'), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)

    labels = []
    ids = []  # Each label has a unique id such as "/m/068hy"
    for i in range(1, len(lines)):
        _, id, label = lines[i]
        ids.append(id)
        labels.append(label)

    return labels, ids


@audioset.capture
def get_ancestor_mapping(folder_name):
    root_dir = os.path.join(get_dataset_dir(), folder_name)
    metadata_dir = os.path.join(root_dir, 'metadata')
    with open(os.path.join(metadata_dir, 'ancestors.json'), 'r') as f:
        ancestors = json.load(f)
    return ancestors


class AudioSetDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, split='train', update_parent_labels=False):

        assert os.path.exists(root_dir), f'Parameter \'root_dir\' is invalid. {root_dir} does not exist.'
        assert split in SPLITS, f'Parameter \'split\' must be in {SPLITS}.'

        self.metadata_dir = os.path.join(root_dir, 'metadata')
        self.update_parent_labels = update_parent_labels
        self.split = split

        ###
        # load the labels, create label index mapping
        ###
        self.labels, self.ids = get_class_labels_ids()
        self.ancestors = get_ancestor_mapping()
        self.lb_to_ix = {label: i for i, label in enumerate(self.labels)}
        self.id_to_ix = {id: i for i, id in enumerate(self.ids)}
        self.ix_to_lb = {i: label for i, label in enumerate(self.labels)}

        ###
        # load file list
        ###
        self.ytids, self.parts, self.targets = [], [], []

        if split == 'train':
            self.audio_dir = os.path.join(root_dir, 'audios', 'unbalanced_train_segments')
            for i in tqdm.tqdm(range(41), desc='Loading meta data'):
                file_path = os.path.join(self.metadata_dir, 'unbalanced_partial_csvs',
                                         f'unbalanced_train_segments_part{i:02d}.csv')
                self.append_csv_to_lists(file_path, part=i)
        elif split == 'balanced':
            self.audio_dir = os.path.join(root_dir, 'audios', 'balanced_train_segments')
            file_path = os.path.join(self.metadata_dir, 'balanced_train_segments.csv')
            self.append_csv_to_lists(file_path)
        elif split == 'evaluation':
            self.audio_dir = os.path.join(root_dir, 'audios', 'eval_segments')
            file_path = os.path.join(self.metadata_dir, 'eval_segments.csv')
            self.append_csv_to_lists(file_path)

    def get_updated_csv_rows(self, file_path, part):
        cleaned_csv_file = file_path[:-4] + '_cleaned.csv'

        if not os.path.exists(cleaned_csv_file):
            with open(file_path, 'r') as f:
                lines = f.readlines()
                lines = list(lines)[3:]

            with open(cleaned_csv_file, 'w') as f:
                for line in lines:
                    line = line.split(', ')
                    if self.split == 'train':
                        path = os.path.join(self.audio_dir, f'unbalanced_train_segments_part{part:02d}',
                                            'Y' + line[0] + '.wav')
                    else:
                        path = os.path.join(self.audio_dir, 'Y' + line[0] + '.wav')
                    if os.path.exists(path) or os.path.exists(path[:-3] + 'mp3'):
                        f.write(', '.join(line))

        with open(cleaned_csv_file, 'r') as f:
            lines = f.readlines()
            lines = list(lines)

        return lines

    def append_csv_to_lists(self, file_path, part=None):
        lines = self.get_updated_csv_rows(file_path, part)
        for line in lines:
            line = line.split(', ')
            self.ytids.append('Y' + line[0])
            self.parts.append(part)
            label_ids = line[3].split('"')[1].split(',')
            target = np.zeros(len(self.lb_to_ix), dtype=np.bool_)
            for id in label_ids:
                ix = self.id_to_ix[id]
                target[ix] = 1
                if self.update_parent_labels:
                    for a in self.ancestors[id]:
                        ix = self.id_to_ix[a]
                        target[ix] = 1
            self.targets.append(target)

    def __len__(self):
        return len(self.ytids)

    def __getitem__(self, index):
        if self.split == 'train':
            part = self.parts[index]
            path = os.path.join(self.audio_dir, f'unbalanced_train_segments_part{part:02d}', self.ytids[index] + '.wav')
        else:
            path = os.path.join(self.audio_dir, self.ytids[index] + '.wav')

        return {
            'path': path,
            'ytid': self.ytids[index],
            'target': self.targets[index].copy() # copy, just to be save ...
        }

    def __str__(self):
        return f'{type(self)}_{self.split}'
