from typing import Iterable

import torch

from lib.nn.gather import build_optimal_gather_module, build_optimal_single_layer_gather_module_unwrapped
from lib.nn.topological.layers import Ordinals
from lib.nn.weight import build_weights_from_java
from lib.utils import value_to_tensor


def _build_weights(weights: Iterable, out_map: dict[str, int], weights_out: list[torch.Tensor]):
    for weight in weights:
        w_idx = str(weight.index)

        if w_idx not in out_map:
            out_map[w_idx] = len(out_map)

            w_tensor = value_to_tensor(weight.value)
            weights_out.append(w_tensor)


class Linear(torch.nn.Module):
    def __init__(
        self,
        layer_neurons: list,
        neuron_ordinals: Ordinals,
        period: int | None = None,
    ) -> None:
        super().__init__()

        self.neuron_ids = [n.getIndex() for n in layer_neurons]

        self.period = period

        self.gather = build_optimal_gather_module(
            [neuron_ordinals[int(inp.getIndex())] for n in layer_neurons for inp in n.getInputs()],
            period=period,
        )

        self.weight, weight_idx_map = build_weights_from_java(layer_neurons)

        def _iter_all_weights():
            return (w for n in layer_neurons for w in n.getWeights())

        weight_idxs: list[int] = [weight_idx_map[str(w.index)] for w in _iter_all_weights()]
        self.gather_weights = build_optimal_single_layer_gather_module_unwrapped(ordinals=weight_idxs, period=period)

    def forward(self, layer_values: dict[int, torch.Tensor]):
        input_values = self.gather(layer_values)
        w = self.weight()
        w = self.gather_weights(w)
        y = w @ input_values
        return y

    def extra_repr(self):
        return f"period={self.period},"
