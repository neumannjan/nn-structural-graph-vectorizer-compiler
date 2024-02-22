from typing import List, Literal, Protocol

import torch
from torch_scatter import scatter_add, scatter_mean

from lib.nn.utils.utils import ViewWithPeriod

AggregationType = Literal["mean", "sum"]


class SimpleAggregation(torch.nn.Module):
    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    @property
    def is_matching_dimension(self) -> bool:
        return True

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


class ReshapeAggregateLike(Protocol):
    @property
    def is_matching_dimension(self) -> bool:
        ...

    def get_reshape(self) -> ViewWithPeriod:
        """Give ViewWithPeriod if is_matching_dimension == True, else fail."""
        ...

    def get_aggregate(self) -> SimpleAggregation:
        """Give SimpleAggregation if is_matching_dimension == True, else fail."""
        ...


class ReshapeAggregateModuleLike(ReshapeAggregateLike, Protocol):
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        ...


class ReshapeAndAggregate(torch.nn.Module, ReshapeAggregateModuleLike):
    def __init__(self, period: int, agg_type: AggregationType, dim: int = 1) -> None:
        super().__init__()
        self.reshape = ViewWithPeriod(period=period)
        self.aggregate = _build_simple_aggregation(agg_type, dim=dim)

    @property
    def is_matching_dimension(self) -> bool:
        return True

    def get_reshape(self) -> ViewWithPeriod:
        return self.reshape

    def get_aggregate(self) -> SimpleAggregation:
        return self.aggregate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reshape(x)
        x = self.aggregate(x)
        return x


def _get_scatter_func(agg: AggregationType):
    if agg == "mean":
        return scatter_mean
    elif agg == "sum":
        return scatter_add
    else:
        raise ValueError()


class ScatterReduceAggregation(torch.nn.Module, ReshapeAggregateModuleLike):
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

    @property
    def is_matching_dimension(self) -> bool:
        return False

    def get_reshape(self) -> ViewWithPeriod:
        raise ValueError("is_matching_dimension == False!")

    def get_aggregate(self) -> SimpleAggregation:
        raise ValueError("is_matching_dimension == False!")

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.scatter_func(inp, self.index, dim=0)

    def extra_repr(self) -> str:
        return f"type={self.agg_type},"


def build_optimal_reshape_aggregate(agg_type: AggregationType, counts: List[int] | torch.Tensor):
    if not isinstance(counts, torch.Tensor):
        counts = torch.tensor(counts, dtype=torch.int32)

    if (counts[1:] == counts[0]).all():
        return ReshapeAndAggregate(period=int(counts[0].item()), agg_type=agg_type, dim=1)

    return ScatterReduceAggregation(agg_type, counts=counts)
