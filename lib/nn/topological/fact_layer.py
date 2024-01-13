import torch
from lib.utils import atleast_3d_rev, value_to_numpy, value_to_tensor


class FactLayer(torch.nn.Module):
    def __init__(self, layer_neurons: list, assume_facts_same=True) -> None:
        super().__init__()

        self.neuron_ids = [n.getIndex() for n in layer_neurons]

        if assume_facts_same:
            neuron = layer_neurons[0]

            value_np = value_to_numpy(neuron.getRawState().getValue())
            self.value = atleast_3d_rev(torch.tensor(value_np))

            # check
            for n in layer_neurons[1:]:
                assert (value_to_numpy(n.getRawState().getValue()) == value_np).all()
        else:
            self.value = atleast_3d_rev(
                torch.stack([value_to_tensor(n.getRawState().getValue()) for n in layer_neurons])
            )

        # check
        for n in layer_neurons:
            # TODO: ASSUMPTION: is not learnable
            assert not n.hasLearnableValue
            # TODO: ASSUMPTION: weight is 1

    def forward(self, *kargs, **kwargs):
        return self.value


