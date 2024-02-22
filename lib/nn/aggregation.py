from typing import List, Literal

import torch
from torch_scatter import scatter_add, scatter_mean

from lib.nn.utils.utils import ReshapeWithPeriod

AggregationType = Literal["mean", "sum"]


class SimpleAggregation(torch.nn.Module):
    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class Mean(SimpleAggregation):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.mean(dim=self.dim)


class Sum(SimpleAggregation):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.sum(dim=self.dim)


def _build_simple_aggregation(agg_type: AggregationType, dim: int = 1):
    if agg_type == "mean":
        return Mean(dim)
    elif agg_type == "sum":
        return Sum(dim)
    else:
        raise ValueError()


class ReshapeAndAggregate(torch.nn.Module):
    def __init__(self, period: int, agg_type: AggregationType, dim: int = 1) -> None:
        super().__init__()
        self.reshape = ReshapeWithPeriod(period=period)
        self.aggregation = _build_simple_aggregation(agg_type, dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reshape(x)
        x = self.aggregation(x)
        return x

    @property
    def period(self) -> int:
        return self.reshape.period

    @property
    def dim(self) -> int:
        return self.aggregation.dim


def _get_scatter_func(agg: AggregationType):
    if agg == "mean":
        return scatter_mean
    elif agg == "sum":
        return scatter_add
    else:
        raise ValueError()


class ScatterReduceAggregation(torch.nn.Module):
    def __init__(self, agg_type: AggregationType, counts: List[int] | torch.Tensor) -> None:
        super().__init__()
        self.agg_type = agg_type

        if not isinstance(counts, torch.Tensor):
            counts = torch.tensor(counts, dtype=torch.int32)

        self.index = torch.nn.Parameter(
            torch.repeat_interleave(torch.arange(0, counts.shape[0]), repeats=counts),
            requires_grad=False,
        )

        self.scatter_func = _get_scatter_func(agg_type)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.scatter_func(inp, self.index, dim=0)

    def extra_repr(self) -> str:
        return f"type={self.agg_type},"


def build_optimal_aggregation(agg_type: AggregationType, counts: List[int] | torch.Tensor):
    if not isinstance(counts, torch.Tensor):
        counts = torch.tensor(counts, dtype=torch.int32)

    if (counts[1:] == counts[0]).all():
        return ReshapeAndAggregate(period=int(counts[0].item()), agg_type=agg_type, dim=1)

    return ScatterReduceAggregation(agg_type, counts=counts)
