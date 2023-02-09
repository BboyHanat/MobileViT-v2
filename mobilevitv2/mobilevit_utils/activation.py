from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F    # noqa


# class Swish(nn.Module):
#
#     def __init__(self, bata):
#         super(Swish, self).__init__()
#         self._bata = bata
#
#     def forward(self, x):
#         return x*torch.sigmoid(x*self._bata)


class Swish(nn.SiLU):
    """
    Applies the `Swish (also known as SiLU) <https://arxiv.org/abs/1702.03118>`_ function.
    """

    def __init__(self, inplace: Optional[bool] = False, *args, **kwargs) -> None:
        super().__init__(inplace=inplace)
