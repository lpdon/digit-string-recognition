from functools import reduce
from typing import List, Any

import torch


def concat(lists: List[List[Any]]) -> List[Any]:
    return reduce(lambda l1, l2: l1 + l2, lists)


def length_tensor(lists: List[List[Any]]) -> torch.Tensor:
    return torch.Tensor([len(t) for t in lists]).type(torch.long)
