import torch

from lib.engines.torch.model import LayeredInputType


class ConcatRefsModule(torch.nn.Module):
    def __init__(self, names: list[str]) -> None:
        super().__init__()
        self.names = names

    def forward(self, inputs: LayeredInputType):
        xs = [inputs[n] for n in self.names]
        y = torch.concat(xs)
        return y

    def extra_repr(self) -> str:
        return ", ".join(self.names)


class RetrieveRefModule(torch.nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def forward(self, inputs: LayeredInputType):
        out = inputs[self.name]
        return out

    def extra_repr(self) -> str:
        return self.name
