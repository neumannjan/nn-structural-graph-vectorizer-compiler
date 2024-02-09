from collections import defaultdict
from typing import Iterable

import torch
from tqdm.auto import tqdm

from lib.nn.gather import build_optimal_gather_module
from lib.nn.topological.layers import Ordinals
from lib.nn.weight import Weight, create_weight
from lib.utils import value_to_tensor


def _group_matching_ordinals_together(weight_indices: Iterable[int]) -> list[tuple[int, list[int]]]:
    weight_indices_to_input_ordinals: dict[int, list[int]] = defaultdict(lambda: [])

    for ord, w_idx in enumerate(weight_indices):
        weight_indices_to_input_ordinals[w_idx].append(ord)

    return list(weight_indices_to_input_ordinals.items())


def _group_matching_consecutive_ordinals_together(weight_indices: Iterable[int]) -> list[tuple[int, list[int]]]:
    # TODO
    raise NotImplementedError()


def _no_group(weight_indices: Iterable[int]) -> list[tuple[int, list[int]]]:
    return [(idx, [ord]) for ord, idx in enumerate(weight_indices)]


class WeightedRuleLayer(torch.nn.Module):
    def __init__(
        self,
        layer_neurons: list,
        neuron_ordinals: Ordinals,
        check_same_weights_assumption=True,
    ) -> None:
        super().__init__()

        neuron = layer_neurons[0]

        weight_map = {int(w.index): w for w in neuron.getWeights()}
        weight_indices = [int(w.index) for w in neuron.getWeights()]

        print(weight_indices)
        if check_same_weights_assumption:
            for n in tqdm(layer_neurons[1:], desc="Verifying neurons"):
                this_widx = [int(w.index) for w in n.getWeights()]
                assert all(
                    (a == int(b.index) for a, b in zip(weight_indices, n.getWeights()))
                ), f"{weight_indices} != {this_widx}"

        weight_indices_to_input_ordinals = _no_group(weight_indices)

        self.len_weights = len(weight_indices)
        self.weights = torch.nn.ModuleList()
        self.gathers = torch.nn.ModuleList()

        for w_idx, input_ords in weight_indices_to_input_ordinals:
            w = weight_map[w_idx]
            self.weights.append(
                create_weight(value_to_tensor(w.value), is_learnable=w.isLearnable())
            )

            self.gathers.append(
                build_optimal_gather_module(
                    [neuron_ordinals[int(n.getInputs()[i].getIndex())] for i in input_ords for n in layer_neurons]
                )
            )

    def forward(self, layer_values: dict[int, torch.Tensor]):
        ys = []

        for i in range(self.len_weights):
            inp = self.gathers[i](layer_values)  # TODO: replace with a single transposition
            w: Weight = self.weights[i]
            y_this = w.apply_to(inp)
            ys.append(y_this)

        y = torch.stack(torch.broadcast_tensors(*ys))

        # TODO: parameterize
        y = torch.sum(y, 0)
        y = torch.tanh(y)
        return y
