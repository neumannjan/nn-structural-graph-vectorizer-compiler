import numpy as np
import torch

from lib.nn.topological.layers import LayerOrdinal


class GatherModule(torch.nn.Module):
    def __init__(self, input_layer_ordinal_pairs: list[LayerOrdinal], allow_merge_on_all_inputs_same: bool) -> None:
        super().__init__()

        if allow_merge_on_all_inputs_same:
            all_inputs_the_same = all((input_layer_ordinal_pairs[0] == p for p in input_layer_ordinal_pairs[1:]))

            if all_inputs_the_same:
                input_layer_ordinal_pairs = [input_layer_ordinal_pairs[0]]

        self.input_layer_ordinal_pairs = input_layer_ordinal_pairs

        self.layers = sorted(set((l for l, _ in input_layer_ordinal_pairs)), reverse=True)
        layers_map = {l: i for i, l in enumerate(self.layers)}

        if len(self.layers) == 1:
            per_layer_ordinals: dict[int, list[int]] = {self.layers[0]: [o for _, o in input_layer_ordinal_pairs]}
        else:
            per_layer_ordinals: dict[int, list[int]] = {
                layer: sorted(set((o for l, o in input_layer_ordinal_pairs if l == layer))) for layer in self.layers
            }

            layer_sizes = [len(per_layer_ordinals[layer]) for layer in self.layers]
            layer_prefixes = np.concatenate([[0], np.cumsum(layer_sizes)[:-1]])

            per_layer_ordinals2_map: dict[int, dict[int, int]] = {
                l: {o: i for i, o in enumerate(per_layer_ordinals[l])} for l in self.layers
            }

            concatenated_ordinals = [
                layer_prefixes[layers_map[l]] + per_layer_ordinals2_map[l][o] for l, o in input_layer_ordinal_pairs
            ]

            self.concatenated_ordinals = torch.tensor(concatenated_ordinals)
        self.per_layer_ordinals = {l: torch.tensor(ordinals) for l, ordinals in per_layer_ordinals.items()}

    def forward(self, layer_values: dict[int, torch.Tensor]):
        # if there is only one layer, skip the additional gather
        if len(self.layers) == 1:
            layer = self.layers[0]
            return torch.index_select(layer_values[layer], 0, self.per_layer_ordinals[layer])

        layer_inputs_needed = [
            torch.index_select(layer_values[layer], 0, self.per_layer_ordinals[layer]) for layer in self.layers
        ]
        layer_shape_hull = [
            -1,
            max((v.shape[1] for v in layer_inputs_needed)),
            max((v.shape[2] for v in layer_inputs_needed)),
        ]
        layer_inputs_needed = [v.expand(*layer_shape_hull) for v in layer_inputs_needed]
        layer_inputs_needed = torch.concatenate(layer_inputs_needed)

        out = torch.index_select(layer_inputs_needed, 0, self.concatenated_ordinals)
        return out


