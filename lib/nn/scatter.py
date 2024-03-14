from typing import Literal, Sequence

import numpy as np
import torch
from torch_scatter import scatter, segment_coo, segment_csr

ReductionType = Literal["mean", "sum", "min", "max"]


class _ScatterBase(torch.nn.Module):
    def __init__(self, index: torch.Tensor, reduce: ReductionType) -> None:
        super().__init__()
        self.index = torch.nn.Parameter(index, requires_grad=False)
        self.reduce = reduce

    def extra_repr(self) -> str:
        return f"reduce={self.reduce}"


class Scatter(_ScatterBase):
    def forward(self, x: torch.Tensor):
        return scatter(x, index=self.index, dim=0, reduce=self.reduce)


class SegmentCOO(_ScatterBase):
    def forward(self, x: torch.Tensor):
        return segment_coo(x, index=self.index, reduce=self.reduce)


class SegmentCSR(_ScatterBase):
    def __init__(self, index: torch.Tensor, indptr: torch.Tensor, reduce: ReductionType) -> None:
        super().__init__(index, reduce)
        self.indptr = torch.nn.Parameter(indptr, requires_grad=False)

    def forward(self, x: torch.Tensor):
        return segment_csr(x, indptr=self.indptr, reduce=self.reduce)


def coo_to_csr_dim0(index: torch.Tensor, do_assert: bool = True) -> torch.Tensor:
    index_diff = index[1:] - index[:-1]

    if do_assert:
        assert (index_diff >= 0).all().item()

    out1 = index_diff.nonzero().squeeze(-1) + 1
    out0 = torch.zeros_like(out1[0]).unsqueeze(0)
    out2 = torch.zeros_like(out1[0]).unsqueeze(0) + index.shape[0]
    out = torch.concat([out0, out1, out2], dim=0)
    return out


def build_optimal_scatter(
    index: torch.Tensor | Sequence[int] | np.ndarray, reduce: ReductionType, allow_non_builtin_torch_ops: bool
):
    if not isinstance(index, torch.Tensor):
        index = torch.tensor(index, dtype=torch.int32)

    if allow_non_builtin_torch_ops:
        if ((index[1:] - index[:-1]) >= 0).all().item():
            indptr = coo_to_csr_dim0(index, do_assert=False)
            return SegmentCSR(index, indptr, reduce)

    return Scatter(index, reduce)
