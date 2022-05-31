from utils import safe_create

import json
import os
import shutil

import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import DatasetFolder, ImageFolder


class PredictFolder(ImageFolder):
    ''' overrides dataset folder '''
    def find_classes(self, directory):
        return (['none'], {'none': 0})

    def make_dataset(self,
            directory, class_to_idx, extensions, is_valid_file=lambda _: True):
        ''' Generates a list of samples of a form (path_to_sample, class) '''
        _out = []
        for path in os.listdir(directory):
            if is_valid_file(path):
                _out.append((os.path.join(directory, path), 0))
        return _out

    def __getitem__(self, index):
        ''' overrides get item '''
        images, _ = super().__getitem__(index)
        return (images, self.samples[index][0])


def _generate_dataset(
    path,
    reload=True,
    masks=False,
    is_valid_file=lambda _: True,
    transforms=None):
    ''' generate directory and returns ImageFolder '''
    with open(os.path.join(path, 'classes.json'), 'r') as fid:
        classes = json.load(fid)

    loader_dir = os.path.join(path, '_loadable')
    safe_create(loader_dir, replace=reload)

    for name, label in classes.items():
        labeldir = os.path.join(loader_dir, str(label))
        safe_create(labeldir, replace=False, allowed=True)
        shutil.copyfile(
            os.path.join(path, 'masks' if masks else 'images', name),
            os.path.join(labeldir, name))
    return ImageFolder(
        root=loader_dir,
        is_valid_file=is_valid_file,
        transform=transforms)


def generate_dataset(
    path,
    reload=False,
    masks=False,
    nattempts=2,
    is_valid_file=lambda _: True,
    transforms=None):
    ''' outer function with error handling '''
    try:
        return _generate_dataset(path, reload=reload, masks=masks, is_valid_file=is_valid_file, transforms=transforms)
    except Exception as e:
        loader_dir = os.path.join(path, '_loadable')
        if os.path.isdir(loader_dir):
            shutil.rmtree(loader_dir)
        if nattempts > 0:
            return generate_dataset(
                path,
                reload=reload,
                masks=masks,
                nattempts=nattempts-1,
                is_valid_file=is_valid_file,
                transforms=transforms)
        else:
            print(
                'tried and failed to make dataset {} times'.format(nattempts))
            raise e


def generate_predict_dataset(
    path,
    is_valid_file=lambda _: True,
    transforms=None):
    ''' generate prediction folder '''
    return PredictFolder(
        root=path,
        is_valid_file=is_valid_file,
        transform=transforms)



def get_train_test_samplers(nb_data, ratio=(80, 20), seed=None):
    ''' splits a dataset into test train '''
    if seed is not None:
        np.random.seed(seed)

    if isinstance(ratio, (list, tuple)):
        rate = ratio[0] / (ratio[0] + ratio[1])
    else:
        rate = ratio if ratio < 1 else (ratio / 100)

    indices = list(range(nb_data))
    np.random.shuffle(indices)

    split = int(np.floor(rate * nb_data))

    train_idx, test_idx = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(indices=train_idx)
    test_sampler = SubsetRandomSampler(indices=test_idx)

    return train_sampler, test_sampler
