import numpy as np
import torch

from lib.nn.topological.layers import LayerOrdinal


class TakeValue(torch.nn.Module):
    """
    Gather for single-layer, single-input. Is also used for single-layer when all inputs are known to be identical.

    Simply returns the single value.
    """

    def __init__(self, input_layer: int, ordinal: int) -> None:
        super().__init__()
        self.layer = input_layer
        self.ordinal = ordinal

    def forward(self, layer_values: dict[int, torch.Tensor]):
        out = layer_values[self.layer][self.ordinal]
        return out.reshape([1, *out.shape])

    def extra_repr(self) -> str:
        return f"layer={self.layer}, ordinal={self.ordinal}"


class TakeLayerSlice(torch.nn.Module):
    """
    Gather for inputs coming from a single layer, used when slicing is equivalent to gathering.

    Simply takes a slice from the layer and returns it as-is. Used when the ordinals are equal to `i+range(j-i)`.
    """

    def __init__(self, input_layer: int, start: int, end: int) -> None:
        super().__init__()
        self.layer = input_layer
        self.start = start
        self.end = end

    def forward(self, layer_values: dict[int, torch.Tensor]):
        return layer_values[self.layer][self.start: self.end]

    def extra_repr(self) -> str:
        return f"layer={self.layer}, start={self.start}, end={self.end}"


class SingleLayerGather(torch.nn.Module):
    """Gather for inputs coming from a single layer."""

    def __init__(self, input_layer: int, ordinals: list[int]) -> None:
        super().__init__()
        self.layer = input_layer
        self.ordinals = torch.nn.Parameter(torch.tensor(ordinals, dtype=torch.int32), requires_grad=False)

    def forward(self, layer_values: dict[int, torch.Tensor]):
        input = layer_values[self.layer]
        with torch.profiler.record_function('GATHER_SINGLE_INDEX_SELECT'):
            return torch.index_select(input, 0, self.ordinals)

    def extra_repr(self) -> str:
        return f"layer={self.layer}, ordinals=(list of size {self.ordinals.shape[0]})"


def build_optimal_single_layer_gather_module(input_layer: int, ordinals: list[int]):
    """Build the optimal gather network module when inputs are guaranteed to come from a single layer."""
    all_inputs_the_same = all((ordinals[0] == o for o in ordinals[1:]))

    if all_inputs_the_same:
        return TakeValue(input_layer, ordinals[0])

    all_ordinals_differ_by_1 = all((b - a == 1 for a, b in zip(ordinals[:-1], ordinals[1:])))

    if all_ordinals_differ_by_1:
        return TakeLayerSlice(input_layer, ordinals[0], ordinals[-1] + 1)

    return SingleLayerGather(input_layer, ordinals)


class MultiLayerGather(torch.nn.Module):
    """
    Gather for inputs coming from multiple layers.

    First performs individual gathers for each input layer. Then concatenates the result to a single tensor and performs
    the final gather.
    """

    def __init__(self, input_layer_ordinal_pairs: list[LayerOrdinal]) -> None:
        super().__init__()
        self.input_layer_ordinal_pairs = input_layer_ordinal_pairs

        self.layers = sorted(set((l for l, _ in input_layer_ordinal_pairs)), reverse=True)
        layers_map = {l: i for i, l in enumerate(self.layers)}

        ### setup single-layer gathers ###

        per_layer_ordinals: dict[int, list[int]] = {
            layer: sorted(set((o for l, o in input_layer_ordinal_pairs if l == layer))) for layer in self.layers
        }

        self.layer_gathers = torch.nn.ModuleDict(
            {str(l): build_optimal_single_layer_gather_module(l, per_layer_ordinals[l]) for l in self.layers}
        )

        ### setup final gather ###

        layer_sizes = [len(per_layer_ordinals[layer]) for layer in self.layers]
        layer_prefixes = np.concatenate([[0], np.cumsum(layer_sizes)[:-1]])

        per_layer_ordinals2_map: dict[int, dict[int, int]] = {
            l: {o: i for i, o in enumerate(per_layer_ordinals[l])} for l in self.layers
        }

        concatenated_ordinals = [
            layer_prefixes[layers_map[l]] + per_layer_ordinals2_map[l][o] for l, o in input_layer_ordinal_pairs
        ]

        self.final_gather = build_optimal_single_layer_gather_module(-1, concatenated_ordinals)

    def forward(self, layer_values: dict[int, torch.Tensor]):
        with torch.profiler.record_function('GATHER_MULTI_RUN_INDIVIDUAL_LAYERS'):
            layer_inputs_needed = [self.layer_gathers[str(layer)](layer_values) for layer in self.layers]

        with torch.profiler.record_function('GATHER_MULTI_EXPAND_CONCAT'):
            layer_shape_hull = [
                -1,
                max((v.shape[1] for v in layer_inputs_needed)),
                max((v.shape[2] for v in layer_inputs_needed)),
            ]
            layer_inputs_needed = [v.expand(*layer_shape_hull) for v in layer_inputs_needed]
            layer_inputs_needed = torch.concatenate(layer_inputs_needed)

        with torch.profiler.record_function('GATHER_MULTI_RUN_FINAL'):
            out = self.final_gather({-1: layer_inputs_needed})
        return out


def build_optimal_gather_module(input_layer_ordinal_pairs: list[LayerOrdinal]):
    layer0, _ = input_layer_ordinal_pairs[0]
    is_single_layer = all((layer0 == l for l, _ in input_layer_ordinal_pairs[1:]))

    if is_single_layer:
        return build_optimal_single_layer_gather_module(layer0, [o for _, o in input_layer_ordinal_pairs])

    return MultiLayerGather(input_layer_ordinal_pairs)
