from typing import Tuple, List, Dict
from warnings import warn

import torch.utils.data as data
from torch.utils.data import Subset

from PIL import Image

import os
import os.path


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
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        samples (list): List of (sample path, subset_index) tuples
    """

    def __init__(self, root, loader=default_loader, transform=None, target_transform=None):
        samples, subset_map = discover_dataset(root)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root))

        self.root = root
        self.loader = loader

        self.samples = samples
        self.subsets = self.create_subsets(subset_map)

        self.transform = transform
        self.target_transform = target_transform

    def create_subsets(self, subset_map: Dict[str, List[str]]) -> Dict[str, Subset]:
        subsets = {}
        for subset_name, indices in subset_map.items():
            subset = Subset(self, indices)
            subsets[subset_name] = subset
        return subsets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
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
