from typing import Type

import torch
import torch.nn.functional as F

from lib.model.ops import ReductionDef


class _DimReduceModule(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        raise NotImplementedError()

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class MeanReduceModule(_DimReduceModule):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim)


class SumReduceModule(_DimReduceModule):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=self.dim)


class SoftmaxReduceModule(_DimReduceModule):
    def forward(self, x: torch.Tensor):
        return F.softmax(x, dim=self.dim)


class MinReduceModule(_DimReduceModule):
    def forward(self, x: torch.Tensor):
        return x.min(dim=self.dim)


class MaxReduceModule(_DimReduceModule):
    def forward(self, x: torch.Tensor):
        return x.max(dim=self.dim)


_MODULES: dict[ReductionDef, Type[_DimReduceModule]] = {
    "mean": MeanReduceModule,
    "avg": MeanReduceModule,
    "average": MeanReduceModule,
    "minimum": MinReduceModule,
    "min": MinReduceModule,
    "sum": SumReduceModule,
    "max": MaxReduceModule,
    "maximum": MaxReduceModule,
}


def build_dim_reduce_module(dim: int, reduce: ReductionDef):
    return _MODULES[reduce](dim)
