from typing import Sequence

import torch


class ViewModule(torch.nn.Module):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor):
        return x.view(self.shape)

    def extra_repr(self) -> str:
        return f"[{', '.join((str(v) for v in self.shape))}]"
