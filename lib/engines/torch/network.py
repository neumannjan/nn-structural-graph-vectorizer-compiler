from typing import Iterable, Type

import torch

from lib.engines.torch.linear import LinearModule
from lib.engines.torch.model import LayeredInputType


class NetworkParams(torch.nn.Module):
    def __init__(
        self,
        params: torch.nn.ParameterDict,
    ) -> None:
        super().__init__()

        self.params = params

    def forward(self) -> LayeredInputType:
        out: LayeredInputType = {k: v for k, v in self.params.items()}
        return out


_NONSEQUENTIAL_MODULES: tuple[Type[torch.nn.Module]] = (LinearModule,)


class LayerModule(torch.nn.Module):
    def __new__(cls, *kargs, debug: bool = False, **kwargs):
        if debug:
            return super().__new__(_LayerModuleWithDebug)  # pyright: ignore

        return super().__new__(cls)

    def __init__(self, modules: Iterable[torch.nn.Module], out_key: str, *, debug: bool = False) -> None:
        super().__init__()
        self.the_modules = torch.nn.ModuleList(modules)
        self.is_nonseq = [isinstance(m, _NONSEQUENTIAL_MODULES) for m in self.the_modules]
        self.out_key = out_key

    def forward(self, inputs: LayeredInputType):
        x = inputs

        for is_nonseq, module in zip(self.is_nonseq, self.the_modules):
            if is_nonseq:
                x = module(x, inputs)
            else:
                x = module(x)

        inputs[self.out_key] = x  # pyright: ignore
        return inputs

    def extra_repr(self) -> str:
        return f"out_key: {self.out_key},"


class _LayerModuleWithDebug(LayerModule):
    def forward(self, inputs: LayeredInputType):
        x = inputs

        for i, (is_nonseq, module) in enumerate(zip(self.is_nonseq, self.the_modules)):
            try:
                if is_nonseq:
                    x = module(x, inputs)
                else:
                    x = module(x)
            except Exception as e:
                raise Exception(f"Exception in layer {self.out_key}, module no. {i}.") from e

        inputs[self.out_key] = x  # pyright: ignore
        return inputs


class NetworkModule(torch.nn.Module):
    def __init__(self, params_module: NetworkParams, batch_modules: torch.nn.ModuleList) -> None:
        super().__init__()
        self.params_module = params_module
        self.batch_modules = batch_modules

    def forward(self, batch: int = 0):
        inputs = self.params_module()
        return self.batch_modules[batch](inputs)
