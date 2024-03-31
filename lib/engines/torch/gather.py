from typing import Sequence

import torch


class TakeValueModule(torch.nn.Module):
    def __init__(self, ordinal: int) -> None:
        super().__init__()
        self.ordinal = ordinal

    def forward(self, x: torch.Tensor):
        return x[self.ordinal].unsqueeze(0)

    def extra_repr(self) -> str:
        return str(self.ordinal)


class SliceValuesModule(torch.nn.Module):
    def __init__(self, start: int, end: int, step: int) -> None:
        super().__init__()
        self.start = start
        self.end = end
        self.step = step

    def forward(self, x: torch.Tensor):
        return x[self.start : self.end : self.step]

    def extra_repr(self) -> str:
        if self.step == 1:
            return f"{self.start} : {self.end}"
        else:
            return f"{self.start} : {self.end} : {self.step}"


class GenericGatherModule(torch.nn.Module):
    def __init__(self, ordinals: Sequence[int]) -> None:
        super().__init__()
        self.ordinals = torch.nn.Parameter(torch.tensor(ordinals, dtype=torch.int32), requires_grad=False)

    def forward(self, x: torch.Tensor):
        return torch.index_select(x, 0, self.ordinals)

    def extra_repr(self) -> str:
        n = 3

        items = ", ".join((str(v) for v in self.ordinals[:n].detach().cpu().tolist()))

        if len(self.ordinals) <= n:
            return f"[{items}]"

        return f"[{items}, ... (size: {len(self.ordinals)})]"


class RepeatModule(torch.nn.Module):
    def __init__(self, repeats: int, total_length: int) -> None:
        super().__init__()
        self.repeats = repeats
        self.total_length = total_length

    def forward(self, x: torch.Tensor):
        x = x.repeat(self.repeats, *([1] * (x.dim() - 1)))
        x = x[: self.total_length]
        return x

    def extra_repr(self) -> str:
        return f"repeats={self.repeats}, total_length={self.total_length}"
