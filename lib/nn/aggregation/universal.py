from typing import List, Protocol, Union

import torch
from torch.jit import unused

from lib.nn.aggregation.fixed_count import FixedCountAggregation, build_fixed_count_aggregate
from lib.nn.aggregation.scatter import build_optimal_scatter_aggregate
from lib.nn.definitions.ops import AggregationDef
from lib.nn.gather import ViewWithPeriod
from lib.nn.scatter import Scatter, SegmentCOO, SegmentCSR
from lib.nn.utils import ShapeTransformable


class ReshapeAggregateLike(ShapeTransformable, Protocol):
    @property
    def is_matching_dimension(self) -> bool:
        ...

    def get_reshape(self) -> ViewWithPeriod:
        """Give ViewWithPeriod if is_matching_dimension == True, else fail."""
        ...

    def get_aggregate(self) -> FixedCountAggregation:
        """Give SimpleAggregation if is_matching_dimension == True, else fail."""
        ...


class ReshapeAggregateModuleLike(ReshapeAggregateLike, Protocol):
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        ...


class ViewAndAggregate(torch.nn.Module, ReshapeAggregateModuleLike):
    def __init__(self, view: ViewWithPeriod, aggregate: FixedCountAggregation) -> None:
        super().__init__()
        self.reshape = view
        self.aggregate = aggregate

    @unused
    def compute_output_shape(self, shape_like) -> list[int]:
        shape_like = self.reshape.compute_output_shape(shape_like)
        shape_like = self.aggregate.compute_output_shape(shape_like)
        return shape_like

    @property
    def is_matching_dimension(self) -> bool:
        return True

    def get_reshape(self) -> ViewWithPeriod:
        return self.reshape

    def get_aggregate(self) -> FixedCountAggregation:
        return self.aggregate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reshape(x)
        x = self.aggregate(x)
        return x


class ScatterAggregate(torch.nn.Module, ReshapeAggregateModuleLike):
    def __init__(self, scatter: Scatter | SegmentCOO | SegmentCSR) -> None:
        super().__init__()
        self.delegate = scatter

    @unused
    def compute_output_shape(self, shape_like) -> list[int]:
        return self.delegate.compute_output_shape(shape_like)

    def forward(self, x):
        return self.delegate(x)

    @property
    def is_matching_dimension(self) -> bool:
        return False

    def get_reshape(self) -> ViewWithPeriod:
        raise ValueError("is_matching_dimension == False!")

    def get_aggregate(self) -> FixedCountAggregation:
        raise ValueError("is_matching_dimension == False!")

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}({repr(self.delegate)})"

    def __getattr__(self, name: str) -> Union[torch.Tensor, torch.nn.Module]:
        delegate = super().__getattr__("delegate")

        if name == "delegate":
            return delegate

        return delegate.__getattr__(name)


def build_optimal_reshape_aggregate(
    aggregation: AggregationDef, counts: List[int] | torch.Tensor, allow_non_builtin_torch_ops: bool
):
    if not isinstance(counts, torch.Tensor):
        counts = torch.tensor(counts, dtype=torch.int32)

    if (counts[1:] == counts[0]).all():
        period = int(counts[0].item())
        return ViewAndAggregate(
            view=ViewWithPeriod(period=period),
            aggregate=build_fixed_count_aggregate(aggregation=aggregation, dim=1),
        )

    return ScatterAggregate(
        build_optimal_scatter_aggregate(
            counts, aggregation=aggregation, allow_non_builtin_torch_ops=allow_non_builtin_torch_ops
        )
    )
