from typing import List, Literal, Protocol, Union

import torch

from lib.nn.gather import ViewWithPeriod
from lib.nn.scatter import ReductionType, build_optimal_scatter

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


class ViewAndAggregate(torch.nn.Module, ReshapeAggregateModuleLike):
    def __init__(self, input_length: int, period: int, agg_type: AggregationType, dim: int = 1) -> None:
        super().__init__()
        self.reshape = ViewWithPeriod(input_length=input_length, period=period)
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


class ScatterAggregate(torch.nn.Module, ReshapeAggregateModuleLike):
    def __init__(self, scatter: torch.nn.Module) -> None:
        super().__init__()
        self.delegate = scatter

    def forward(self, x):
        return self.delegate(x)

    @property
    def is_matching_dimension(self) -> bool:
        return False

    def get_reshape(self) -> ViewWithPeriod:
        raise ValueError("is_matching_dimension == False!")

    def get_aggregate(self) -> SimpleAggregation:
        raise ValueError("is_matching_dimension == False!")

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}({repr(self.delegate)})"

    def __getattr__(self, name: str) -> Union[torch.Tensor, torch.nn.Module]:
        delegate = super().__getattr__("delegate")

        if name == "delegate":
            return delegate

        return delegate.__getattr__(name)


def _build_scatter_reduce_aggregation(
    counts: list[int] | torch.Tensor, reduce: ReductionType, allow_non_builtin_torch_ops: bool
):
    if not isinstance(counts, torch.Tensor):
        counts = torch.tensor(counts, dtype=torch.int32)

    index = torch.repeat_interleave(torch.arange(0, counts.shape[0]), repeats=counts)
    return ScatterAggregate(
        build_optimal_scatter(index=index, reduce=reduce, allow_non_builtin_torch_ops=allow_non_builtin_torch_ops)
    )


def build_optimal_reshape_aggregate(
    agg_type: AggregationType, counts: List[int] | torch.Tensor, allow_non_builtin_torch_ops: bool
):
    if not isinstance(counts, torch.Tensor):
        counts = torch.tensor(counts, dtype=torch.int32)

    if (counts[1:] == counts[0]).all():
        period = int(counts[0].item())
        return ViewAndAggregate(input_length=period * counts.shape[0], period=period, agg_type=agg_type, dim=1)

    return _build_scatter_reduce_aggregation(
        counts, reduce=agg_type, allow_non_builtin_torch_ops=allow_non_builtin_torch_ops
    )
