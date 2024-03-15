from typing import get_args as t_get_args

import torch

from lib.nn.definitions.ops import AggregationDef
from lib.nn.scatter import ReductionDef, build_optimal_scatter

_REDUCE_DEFS: set[ReductionDef] = set(t_get_args(ReductionDef))


def build_optimal_scatter_aggregation(
    counts: list[int] | torch.Tensor, aggregation: AggregationDef, allow_non_builtin_torch_ops: bool
):
    if not isinstance(counts, torch.Tensor):
        counts = torch.tensor(counts, dtype=torch.int32)

    index = torch.repeat_interleave(torch.arange(0, counts.shape[0]), repeats=counts)

    if aggregation not in _REDUCE_DEFS:
        raise NotImplementedError(f"Scatter aggregation is not yet implemented for {aggregation}.")

    return build_optimal_scatter(
        index=index, reduce=aggregation, allow_non_builtin_torch_ops=allow_non_builtin_torch_ops
    )
