import csv
from functools import reduce
from typing import List, Any, Dict
from warnings import warn

import torch


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
