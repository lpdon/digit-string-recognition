import os
import os.path
from typing import Tuple, List, Dict
from warnings import warn

import numpy as np
import torch.utils.data as data
from PIL import Image
from torch.utils.data import Subset, Dataset


def train_val_datasets(dataset: Dataset, val_split: float = 0.5, shuffle: bool = True) -> Tuple[Dataset, Dataset]:
    """
    Splits dataset at specified ratio. E.g. to create train-val split.
    :param dataset: the source dataset
    :param val_split: the ratio of samples which should be training samples
    :param shuffle: shuffle the indices
    :return: two data subsets which (train, val)
    """
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_split * num_train))

    if shuffle:
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[:split], indices[split:]
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


def discover_dataset(dir: str, verbose: bool = True) -> Tuple[List[Tuple[str, str]], Dict[str, List[str]]]:
    images = []
    subset_map = {}
    dir = os.path.expanduser(dir)
    idx = 0
    for subset_gt_file in sorted(os.listdir(dir)):
        gt_path = os.path.join(dir, subset_gt_file)
        if not os.path.isfile(gt_path) or not gt_path.endswith(".txt"):
            continue
        subset_name = subset_gt_file[:-7]

        # Assert that corresponding folder exists
        d = gt_path[:-6] + "images"
        if not os.path.isdir(d):
            warn("Found gt-file without corresponding folder: " + subset_name)
            continue
        if verbose:
            print("Found subset: " + subset_name)
        indices = []
        with open(gt_path, "r") as gt_f:
            for line in gt_f.readlines():
                im_file, gt = line.split(sep="\t")
                im_path = os.path.join(d, im_file)
                if not os.path.isfile(im_path):
                    warn("Missing image in file system "
                         "which is referenced in gt-file. ({})".format(im_path))
                item = (im_path, gt)
                images.append(item)
                indices.append(idx)
                idx += 1
        if verbose:
            print("Subset had {} files in it.".format(len(indices)))
        subset_map[subset_name] = indices
    return images, subset_map


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


class CAR(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/train_images/xxx.ext
        root/train_images/xxy.ext
        root/train_images/xxz.ext
        root/train_gt.txt

        root/test_images/123.ext
        root/test_images/nsdf3.ext
        root/test_images/asd932_.ext
        root/test_gt.txt
        The folder in which the sample is stored corresponds to its subset.
        The gt file must contain all image file names and ground truths of the subset
        in a 2 column tab-separated text file.

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
                 subset_name_map='auto', train_val_split: float = 0.8):
        samples, subset_to_idx = discover_dataset(root)
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
