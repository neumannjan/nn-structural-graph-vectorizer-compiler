import torch

from lib.nn.sources.source import Neurons
from lib.nn.topological.settings import Settings
from lib.utils import atleast_3d_rev, head_and_rest


class FactLayer(torch.nn.Module):
    def __init__(self, neurons: Neurons, settings: Settings) -> None:
        super().__init__()

        self.neuron_ids = neurons.ids

        if settings.assume_facts_same:
            first_value, values_rest = head_and_rest(neurons.get_values_torch())

            self.value = torch.nn.Parameter(atleast_3d_rev(first_value), requires_grad=False)

            # check
            for value in values_rest:
                assert (value == first_value).all()
        else:
            self.value = torch.nn.Parameter(
                atleast_3d_rev(torch.stack(list(neurons.get_values_torch()))),
                requires_grad=False,
            )

    def forward(self, *kargs, **kwargs):
        return self.value
