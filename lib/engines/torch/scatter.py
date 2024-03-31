from typing import Any, Literal, Protocol, Sequence

import numpy as np
from lib.nn.definitions.ops import ReductionDef
from torch_scatter import (
    scatter_max,
    scatter_mean,
    scatter_min,
    scatter_sum,
    segment_max_csr,
    segment_mean_csr,
    segment_min_csr,
    segment_sum_csr,
)

import torch

_StrictReductionRef = Literal["min", "max", "sum", "mean"]

_TO_STRICT: dict[ReductionDef, _StrictReductionRef] = {
    "mean": "mean",
    "avg": "mean",
    "average": "mean",
    "minimum": "min",
    "min": "min",
    "sum": "sum",
    "max": "max",
    "maximum": "max",
}


def _to_strict(reduce: ReductionDef) -> _StrictReductionRef:
    return _TO_STRICT[reduce]


class _ScatterFunc(Protocol):
    def __call__(
        self,
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = -1,
        out: torch.Tensor | None = None,
        dim_size: int | None = None,
    ) -> Any: ...


class _SegmentCSRFunc(Protocol):
    def __call__(
        self,
        src: torch.Tensor,
        indptr: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> Any: ...


class _ScatterBase(torch.nn.Module):
    def __init__(self, index: torch.Tensor, reduce: ReductionDef) -> None:
        super().__init__()
        self.index = torch.nn.Parameter(index, requires_grad=False)
        self.reduce: _StrictReductionRef = _to_strict(reduce)

    def extra_repr(self) -> str:
        return f"reduce={self.reduce}"


_SCATTER_FUNC: dict[_StrictReductionRef, _ScatterFunc] = {
    "mean": scatter_mean,
    "max": scatter_max,
    "min": scatter_min,
    "sum": scatter_sum,
}


class Scatter(_ScatterBase):
    def __init__(self, index: torch.Tensor, reduce: ReductionDef) -> None:
        super().__init__(index, reduce)
        self.reduce_func = _SCATTER_FUNC[self.reduce]

    def forward(self, x: torch.Tensor):
        return self.reduce_func(x, index=self.index, dim=0)


_SEGMENT_CSR_FUNC: dict[_StrictReductionRef, _SegmentCSRFunc] = {
    "mean": segment_mean_csr,
    "max": segment_max_csr,
    "min": segment_min_csr,
    "sum": segment_sum_csr,
}


class SegmentCSR(_ScatterBase):
    def __init__(self, index: torch.Tensor, indptr: torch.Tensor, reduce: ReductionDef) -> None:
        super().__init__(index, reduce)
        self.indptr = torch.nn.Parameter(indptr, requires_grad=False)
        self.reduce_func = _SEGMENT_CSR_FUNC[self.reduce]

    def forward(self, x: torch.Tensor):
        return self.reduce_func(x, indptr=self.indptr)


def coo_to_csr_dim0(index: torch.Tensor, do_assert: bool = True) -> torch.Tensor:
    index_diff = index[1:] - index[:-1]

    if do_assert:
        assert (index_diff >= 0).all().item()

    out1 = index_diff.nonzero().squeeze(-1) + 1
    out0 = torch.zeros_like(out1[0]).unsqueeze(0)
    out2 = torch.zeros_like(out1[0]).unsqueeze(0) + index.shape[0]
    out = torch.concat([out0, out1, out2], dim=0)
    return out


def counts_to_index(counts: torch.Tensor | np.ndarray | list[int]):
    if not isinstance(counts, torch.Tensor):
        counts = torch.tensor(counts, dtype=torch.int32)

    index = torch.repeat_interleave(torch.arange(0, counts.shape[0]), repeats=counts)
    return index


def build_scatter_module(
    index: torch.Tensor | Sequence[int] | np.ndarray, reduce: ReductionDef, allow_non_builtin_torch_ops: bool
):
    if not isinstance(index, torch.Tensor):
        index = torch.tensor(index, dtype=torch.int32)

    if allow_non_builtin_torch_ops:
        if ((index[1:] - index[:-1]) >= 0).all().item():
            indptr = coo_to_csr_dim0(index, do_assert=False)
            return SegmentCSR(index, indptr, reduce)

    return Scatter(index, reduce)
