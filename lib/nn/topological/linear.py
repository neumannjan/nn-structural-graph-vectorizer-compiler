from typing import Iterable

import torch

from lib.nn.gather import build_optimal_gather_module, build_optimal_single_layer_gather_module_unwrapped
from lib.nn.topological.layers import Ordinals
from lib.nn.weight import concatenate_weights, create_weight
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

        def _iter_all_weights():
            return (w for n in layer_neurons for w in n.getWeights())

        idx_map: dict[str, int] = {}

        weights_learnable: list[torch.Tensor] = []
        weights_nonlearnable: list[torch.Tensor] = []

        _build_weights(filter(lambda w: w.isLearnable(), _iter_all_weights()), idx_map, weights_learnable)
        _build_weights(filter(lambda w: not w.isLearnable(), _iter_all_weights()), idx_map, weights_nonlearnable)

        if len(weights_learnable) > 0:
            # ensure all learnable weights have the same dimensions
            assert all(
                (weights_learnable[0].shape == w.shape for w in weights_learnable[1:])
            ), f"Learnable weights' dimensions do not match: {[w.shape for w in weights_learnable]}"

            # expand all non-learnable weights to the same shape as learnable weights
            weights_nonlearnable = [
                create_weight(t, is_learnable=False).expand_to(weights_learnable[0]) for t in weights_nonlearnable
            ]

            weight_learnable = torch.nn.Parameter(torch.stack(weights_learnable), requires_grad=True)
        else:
            weight_learnable = None

        if len(weights_nonlearnable) > 0:
            weight_nonlearnable = torch.nn.Parameter(torch.stack(weights_nonlearnable), requires_grad=False)
        else:
            weight_nonlearnable = None

        self.weight = concatenate_weights(weight_learnable, weight_nonlearnable)

        weight_idxs: list[int] = [idx_map[str(w.index)] for w in _iter_all_weights()]
        self.gather_weights = build_optimal_single_layer_gather_module_unwrapped(ordinals=weight_idxs, period=period)

    def forward(self, layer_values: dict[int, torch.Tensor]):
        with torch.profiler.record_function('LINEAR_GATHER'):
            input_values = self.gather(layer_values)
        with torch.profiler.record_function('LINEAR_WEIGHTS_GATHER'):
            w = self.gather_weights(self.weight())
        with torch.profiler.record_function('LINEAR_LINEAR'):
            y = w @ input_values
        return y

    def extra_repr(self):
        return f"period={self.period},"
