from typing import Tuple

import numpy as np
import torch.utils.data as data
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, Dataset


def train_val_datasets(dataset: Dataset, val_split: float = 0.5, shuffle: bool = True) -> Tuple[Dataset, Dataset]:
    """
    Splits dataset at specified ratio. E.g. to create train-val split.
    :param dataset: the source dataset
    :param val_split: the ratio of samples which should be training samples
    :param shuffle: shuffle the indices
    :return: two data subsets which (train, val)
    """
    train_idx, valid_idx = train_test_split(np.arange(len(dataset)), test_size=1 - val_split,
                                            train_size=val_split, shuffle=shuffle)

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, valid_idx)
    return train_dataset, val_dataset


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def map_subset_name(subset, subset_name_map):
    if not subset_name_map:
        return subset
    elif subset_name_map == 'auto':
        keys = ["train", "test"]
        for key in keys:
            if key in subset:
                return key
    elif subset in subset_name_map:
        return subset_name_map[subset]
    return subset


class TransformSubset(data.Subset):
    """
    Subset of a dataset at specified indices which also supports input transforms.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        transform (callable): Function which transforms input
        target_transform (callable): Function which transforms target
    """

    def __init__(self, dataset, indices, transform=None, target_transform=None):
        super(TransformSubset, self).__init__(dataset, indices)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        sample, target = super(TransformSubset, self).__getitem__(idx)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
