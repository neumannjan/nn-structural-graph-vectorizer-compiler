from typing import Type

from lib.nn.definitions.ops import TransformationDef

import torch


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


_MAPPING: dict[TransformationDef, Type[torch.nn.Module]] = {
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


def build_transformation(transformation: TransformationDef):
    assert transformation != "identity"
    try:
        return _MAPPING[transformation]()
    except KeyError:
        raise NotImplementedError(transformation)
