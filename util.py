import csv
import os.path as osp
from functools import reduce
from typing import List, Any, Dict
from warnings import warn

import torch
from torchvision.utils import save_image

from car_dataset import CAR_A_STD, CAR_A_MEAN


def concat(lists: List[List[Any]]) -> List[Any]:
    return reduce(lambda l1, l2: l1 + l2, lists)


def length_tensor(lists: List[List[Any]]) -> torch.Tensor:
    return torch.Tensor([len(t) for t in lists]).type(torch.long)


def format_status_line(status_dict: Dict[str, Any]) -> str:
    formatted_dict = {}
    for key, val in status_dict.items():
        if isinstance(val, float):
            fval = "{:10.6f}".format(val)
        else:
            fval = "{:5}".format(val)
        formatted_dict[key] = fval
    status_line = " | ".join(["{}: {}".format(key, value) for key, value in formatted_dict.items()])
    return status_line


def write_to_csv(history_item: Dict[str, Any], log_file: str,
                 write_header: bool = False, append: bool = True) -> None:
    if write_header and append:
        warn("Writing header but appending to file. Header could be not on first line.")
    with open(log_file, 'a' if append else 'w') as csvfile:
        csvwriter = csv.DictWriter(csvfile, history_item.keys(), delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        if write_header:
            csvwriter.writeheader()
        csvwriter.writerow(history_item)


# https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        :param tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        :return Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class ImageWriter:

    def __init__(self, folder, mean=CAR_A_MEAN, std=CAR_A_STD):
        self.unnormalize = UnNormalize(mean, std)
        self.folder = folder

    def write(self, im, name):
        save_image(self.unnormalize(im), osp.join(self.folder, name))
