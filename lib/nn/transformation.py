from typing import Callable

import torch

from lib.nn.definitions.ops import TransformationDef
from lib.nn.utils import Identity


class Sqrt(torch.nn.Module):
    def forward(self, x):
        return torch.sqrt(x)


class Signum(torch.nn.Module):
    def forward(self, x):
        return torch.sign(x)


class Log(torch.nn.Module):
    def forward(self, x):
        return torch.log(x)


class Exp(torch.nn.Module):
    def forward(self, x):
        return torch.exp(x)


_MAPPING: dict[TransformationDef | None, Callable[[], torch.nn.Module]] = {
    None: Identity,
    "identity": Identity,
    "tanh": torch.nn.Tanh,
    "square_root": Sqrt,
    "signum": Signum,
    "sign": Signum,
    "logarithm": Log,
    "log": Log,
    "ln": Log,
    "relu": torch.nn.ReLU,
    "leaky_relu": torch.nn.LeakyReLU,
    "exponentiation": Exp,
    "exp": Exp,
    "sigmoid": torch.nn.Sigmoid,
}


def build_transformation(transformation: TransformationDef | None):
    try:
        return _MAPPING[transformation]()
    except KeyError:
        raise NotImplementedError(transformation)
