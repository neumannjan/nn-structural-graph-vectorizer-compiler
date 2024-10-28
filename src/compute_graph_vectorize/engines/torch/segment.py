from typing import Any, Literal, Protocol

import torch
from torch_scatter import (
    segment_max_csr,
    segment_mean_csr,
    segment_min_csr,
    segment_sum_csr,
)

_StrictReductionRef = Literal["min", "max", "sum", "mean"]


class _SegmentCSRFunc(Protocol):
    def __call__(
        self,
        src: torch.Tensor,
        indptr: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> Any: ...


SEGMENT_CSR_FUNC: dict[_StrictReductionRef, _SegmentCSRFunc] = {
    "mean": segment_mean_csr,
    "max": segment_max_csr,
    "min": segment_min_csr,
    "sum": segment_sum_csr,
}
