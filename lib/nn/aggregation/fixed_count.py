import warnings
from typing import Type

import torch
import torch.nn.functional as F
from torch.jit import unused

from lib.nn.definitions.ops import AggregationDef
from lib.nn.utils import ShapeTransformable


class FixedCountAggregation(torch.nn.Module, ShapeTransformable):
    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @unused
    def compute_output_shape(self, shape: list[int]) -> list[int]:
        return [shape[0], *shape[2:]]

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class MeanAgg(FixedCountAggregation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim)


class SumAgg(FixedCountAggregation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=self.dim)


class Softmax(FixedCountAggregation):
    def forward(self, x: torch.Tensor):
        return F.softmax(x, dim=self.dim)


class MinAgg(FixedCountAggregation):
    def forward(self, x: torch.Tensor):
        return x.min(dim=self.dim)


class MaxAgg(FixedCountAggregation):
    def forward(self, x: torch.Tensor):
        return x.max(dim=self.dim)


class CountAgg(FixedCountAggregation):
    def __init__(self, dim: int = 1) -> None:
        super().__init__(dim)
        warnings.warn("You are using a count aggregation on a fixed no. of inputs. Are you sure about this?")

    def forward(self, x: torch.Tensor):
        return torch.ones_like(x).sum(dim=self.dim)


_FIXED_COUNT: dict[AggregationDef, Type[FixedCountAggregation]] = {
    "mean": MeanAgg,
    "avg": MeanAgg,
    "average": MeanAgg,
    "softmax": Softmax,
    "minimum": MinAgg,
    "min": MinAgg,
    "sum": SumAgg,
    "maximum": MaxAgg,
    "max": MaxAgg,
    "count": CountAgg,
}


def build_fixed_count_aggregate(aggregation: AggregationDef, dim: int = 1):
    try:
        return _FIXED_COUNT[aggregation](dim=dim)
    except KeyError:
        raise NotImplementedError(aggregation)
