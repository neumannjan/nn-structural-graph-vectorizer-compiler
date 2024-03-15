import torch

from lib.nn.sources.base import LayerNeurons
from lib.nn.topological.settings import Settings
from lib.utils import atleast_3d_rev


class FactLayer(torch.nn.Module):
    def __init__(self, neurons: LayerNeurons, settings: Settings) -> None:
        super().__init__()

        self.neuron_ids = neurons.ids

        transformations = neurons.get_transformations()
        # TODO: assumption: FactLayer has no transformation
        for tr in transformations:
            assert tr in (None, "identity")

        self.value = torch.nn.Parameter(
            atleast_3d_rev(torch.stack(list(neurons.get_values_torch()))),
            requires_grad=False,
        )

    def extra_repr(self) -> str:
        return f"len={self.value.shape[0]}"

    def forward(self, *kargs, **kwargs):
        return self.value
