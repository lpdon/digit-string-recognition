import os
import os.path
from typing import Tuple, List, Dict
from warnings import warn

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, Dataset, DataLoader

from dataset import train_val_datasets, pil_loader, accimage_loader, default_loader, map_subset_name, TransformSubset


def discover_dataset(dir: str, verbose: bool = True) -> Tuple[List[Tuple[str, str]], Dict[str, List[str]]]:
    images = []
    subset_map = {}
    root_dir = os.path.expanduser(dir)
    idx = 0

    dirs = ["train", "test"]

    for dir in dirs:
        indices = []
        d = os.path.join(root_dir, dir)
        for im_file in os.listdir(d):
            gt = im_file.split("-")[0]
            im_path = os.path.join(d, im_file)
            item = (im_path, gt)
            images.append(item)
            indices.append(idx)
            idx += 1
            print(item)
        if verbose:
            print("Subset had {} files in it.".format(len(indices)))
        subset_map[dir] = indices

    return images, subset_map
    

class CVL(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/train/gt-id-xxt.ext

        root/test/gt-id-xxt.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        transform (Dict[str, callable], optional):
            A dict from subset_names to functions which transform the input images.
        target_transform (callable, optional):
            A function/transform that takes in a target and returns a transformed version.
        subset_name_map ('auto' or dict[str, str] or None):
            Either a dict which maps the folder to some chosen subset names (e.g. train, test).
            If 'auto' it will be checked if {train, test} is a substring
            of the subset name and will then be used. Subset names not matching this pattern are not touched.
              e.g: a_train -> train
            'auto' works for the standard CAR-A and CAR-B datasets.
            If None is given the subset names are not changed.
        train_val_split (float): Ratio at which to perform train_val_split.
            Must be greater 0 and smaller or equal than 1
            If equal to 1, no split is done.
            If unequal 1, a subset with name 'train' must exist after mapping.
            If it exists, two subsets 'train' and 'val' will be added to this subset.
            'train' subset is overridden.

     Attributes:
        samples (list): List of (sample path, subset_index) tuples
    """

    def __init__(self, root, loader=default_loader, transform=None, target_transform=None,
                 subset_name_map='auto', train_val_split: float = 0.8, verbose: bool = False):
        samples, subset_to_idx = discover_dataset(root, verbose=verbose)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root))

        self.root = root
        self.loader = loader

        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

        self.subsets = self.create_subsets(subset_to_idx, subset_name_map)
        assert 0.0 < train_val_split <= 1.0
        if train_val_split != 1.0:
            assert 'train' in self.subsets
            self.subsets['train'], self.subsets['val'] = train_val_datasets(self.subsets['train'], train_val_split)

    def create_subsets(self, subset_map: Dict[str, List[str]],
                       subset_name_map) -> Dict[str, Subset]:
        subsets = {}
        for subset_name, indices in subset_map.items():
            subset_name = map_subset_name(subset_name, subset_name_map)
            transform = self.transform[subset_name] if self.transform else None
            target_transform = self.target_transform[subset_name] if self.target_transform else None
            subset = TransformSubset(self, indices, transform, target_transform)
            subsets[subset_name] = subset
        return subsets

    def __getitem__(self, index: int) -> Tuple[Image.Image, str]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of total datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        fmt_str += '\n\tSubsets: \n'
        for name, subset in self.subsets.items():
            fmt_str += '\t\t{}: number of datapoints: {}\n'.format(name, len(subset))
        return fmt_str

    def statistics(self) -> str:
        fmt_str = "Max Width: {}\n".format(max([img.width for img, gt in self]))
        fmt_str += "Max Height: {}\n".format(max([img.height for img, gt in self]))
        fmt_str += "Min Width: {}\n".format(min([img.width for img, gt in self]))
        fmt_str += "Min Height: {}\n".format(min([img.height for img, gt in self]))
        fmt_str += "Avg Width: {}\n".format(sum([img.width for img, gt in self]) / float(len(self)))
        fmt_str += "Avg Height: {}\n".format(sum([img.height for img, gt in self]) / float(len(self)))
        fmt_str += "Avg Aspect: {}\n".format(sum([img.width / img.height for img, gt in self]) / float(len(self)))
        return fmt_str

    def mean_and_std(self) -> Tuple[float, float]:
        loader = DataLoader(
            self.subsets['train'],
            batch_size=10,
            num_workers=1,
            shuffle=False
        )
        mean = torch.full((3,), 0.0)
        std = torch.full((3,), 0.0)
        nb_samples = 0.
        for data, gt in loader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples
        mean /= nb_samples
        std /= nb_samples
        return mean, std
