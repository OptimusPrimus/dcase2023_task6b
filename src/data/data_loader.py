import os
from sacred import Ingredient
from torch.utils.data import WeightedRandomSampler, DataLoader, Sampler
from data.augment import RollAudio
import torch
data_loader = Ingredient('data_loader')


@data_loader.config
def config():
    batch_size = 24
    batch_size_eval = 24
    n_workers = 16

    roll_augment = False

    # only used if targets are given
    queue_random_sampling = False


@data_loader.capture
def get_train_data_loader(data_set, batch_size, n_workers, queue_random_sampling, targets=None, roll_augment=False, collate_fun=None, shuffle=True):

    if roll_augment:
        data_set = RollAudio(data_set)

    if targets is None:
        return DataLoader(data_set, batch_size=batch_size, num_workers=n_workers, shuffle=shuffle, collate_fn=collate_fun)
    else:
        raise AttributeError('not expecting targets')


@data_loader.capture
def get_eval_data_loader(data_set, batch_size_eval, n_workers, collate_fun=None, shuffle=False,distributed=False):
    if distributed:
        print("Using distributed sampler")
        num_replicas=int(os.environ['WORLD_SIZE'])
        rank=int(os.environ['NODE_RANK'])
        sampler = torch.utils.data.DistributedSampler(data_set, shuffle=shuffle,num_replicas=num_replicas,rank=rank)
        return DataLoader(data_set, batch_size=batch_size_eval, num_workers=n_workers, sampler=sampler
                          , collate_fn=collate_fun)
    return DataLoader(data_set, batch_size=batch_size_eval, num_workers=n_workers, shuffle=shuffle, collate_fn=collate_fun)

