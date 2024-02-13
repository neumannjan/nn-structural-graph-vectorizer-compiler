from typing import Protocol, Sequence

import torch


class WeightLike(Protocol):
    def apply_to(self, to: torch.Tensor) -> torch.Tensor:
        ...

    def expand_to(self, other: torch.Tensor | torch.Size | tuple[int, ...]) -> torch.Tensor:
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

    def expand_to(self, other: torch.Tensor | torch.Size | tuple[int, ...]) -> torch.Tensor:
        raise ValueError("Not supported for weights other than UnitWeight")

    def extra_repr(self) -> str:
        shape = ",".join(map(str, self.weight.shape))
        return f"{shape}"

    def forward(self) -> torch.Tensor:
        return self.weight


class UnitWeight(Weight, WeightLike):
    def __init__(self) -> None:
        super().__init__(torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False))

    def expand_to(self, other: torch.Tensor | torch.Size | tuple[int, ...]) -> torch.Tensor:
        shape = other.shape if isinstance(other, torch.Tensor) else other

        if len(shape) == 2 and shape[0] == shape[1]:
            return torch.eye(shape[0])

        return torch.ones(shape)

    def apply_to(self, to: torch.Tensor) -> torch.Tensor:
        return to


def create_weight(tensor: torch.Tensor, is_learnable: bool) -> Weight:
    if not is_learnable and (tensor == torch.tensor([1.0], device=tensor.device)).all():
        return UnitWeight()

    return Weight(torch.nn.Parameter(torch.atleast_1d(tensor), requires_grad=is_learnable))


class WeightsConcatenation(torch.nn.Module):
    def __init__(self, weights: Sequence[torch.nn.Parameter]) -> None:
        super().__init__()

        self.weights = weights

    def forward(self) -> torch.Tensor:
        return torch.concatenate(self.weights)


def concatenate_weights(*weights: torch.nn.Parameter | None) -> torch.nn.Module:
    weights_nonnull = [w for w in weights if w is not None]
    if len(weights_nonnull) == 0:
        raise ValueError()
    elif len(weights_nonnull) == 1:
        return Weight(weights_nonnull[0])

    return WeightsConcatenation(weights_nonnull)
