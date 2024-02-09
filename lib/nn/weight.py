from typing import Protocol

import torch


class WeightLike(Protocol):
    def apply_to(self, to: torch.Tensor) -> torch.Tensor:
        ...

    @property
    def value(self) -> torch.nn.Parameter:
        ...


class Weight(torch.nn.Module, WeightLike):
    def __init__(self, weight: torch.nn.Parameter) -> None:
        super().__init__()
        self.weight = weight

    def apply_to(self, to: torch.Tensor) -> torch.Tensor:
        return self.weight @ to

    @property
    def value(self) -> torch.nn.Parameter:
        return self.weight

    def extra_repr(self) -> str:
        shape = ",".join(map(str, self.weight.shape))
        return f"{shape}"


class UnitWeight(Weight, WeightLike):
    def __init__(self) -> None:
        super().__init__(torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False))

    def apply_to(self, to: torch.Tensor) -> torch.Tensor:
        return to


def create_weight(tensor: torch.Tensor, is_learnable: bool) -> Weight:
    if not is_learnable and (tensor == torch.tensor([1.0], device=tensor.device)).all():
        return UnitWeight()

    return Weight(torch.nn.Parameter(torch.atleast_1d(tensor), requires_grad=is_learnable))
