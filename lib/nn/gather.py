import numpy as np
import torch

from lib.nn.topological.layers import LayerOrdinal
from lib.nn.utils.pipes import LayerInputPipe


class TakeValue(torch.nn.Module):
    """
    Gather for single-layer, single-input. Is also used for single-layer when all inputs are known to be identical.

    Simply returns the single value.
    """

    def __init__(self, ordinal: int) -> None:
        super().__init__()
        self.ordinal = ordinal

    def forward(self, layer_input: torch.Tensor):
        return layer_input[self.ordinal].unsqueeze(0)

    def extra_repr(self) -> str:
        return f"ordinal={self.ordinal}"


class TakeLayerSlice(torch.nn.Module):
    """
    Gather for inputs coming from a single layer, used when slicing is equivalent to gathering.

    Simply takes a slice from the layer and returns it as-is. Used when the ordinals are equal to `i+range(j-i)`.
    """

    def __init__(self, start: int, end: int) -> None:
        super().__init__()
        self.start = start
        self.end = end

    def forward(self, layer_input: torch.Tensor):
        return layer_input[self.start : self.end]

    def extra_repr(self) -> str:
        return f"start={self.start}, end={self.end}"


class TakeEachNth(torch.nn.Module):
    def __init__(self, step: int, start: int, end: int) -> None:
        super().__init__()
        self.step = step
        self.start = start
        self.end = end

    def forward(self, layer_input: torch.Tensor):
        return layer_input[self.start:self.end:self.step]

    def extra_repr(self) -> str:
        return f"step={self.step}, start={self.start}, end={self.end}"


class SingleLayerGather(torch.nn.Module):
    """Gather for inputs coming from a single layer."""

    def __init__(self, ordinals: list[int]) -> None:
        super().__init__()
        self.ordinals = torch.nn.Parameter(torch.tensor(ordinals, dtype=torch.int32), requires_grad=False)

    def forward(self, layer_input: torch.Tensor):
        with torch.profiler.record_function('GATHER_SINGLE_INDEX_SELECT'):
            return torch.index_select(layer_input, 0, self.ordinals)

    def extra_repr(self) -> str:
        return f"ordinals=(list of size {self.ordinals.shape[0]})"


def build_optimal_single_layer_gather_module_unwrapped(ordinals: list[int]):
    all_inputs_the_same = all((ordinals[0] == o for o in ordinals[1:]))

    if all_inputs_the_same:
        return TakeValue(ordinals[0])

    step = ordinals[1] - ordinals[0]
    all_ordinals_differ_by_step = all((b - a == step for a, b in zip(ordinals[:-1], ordinals[1:])))

    if all_ordinals_differ_by_step:
        if step == 1:
            return TakeLayerSlice(ordinals[0], ordinals[-1] + 1)

        return TakeEachNth(step=step, start=ordinals[0], end=ordinals[-1] + 1)

    return SingleLayerGather(ordinals)


def build_optimal_single_layer_gather_module(input_layer: int, ordinals: list[int]):
    """Build the optimal gather network module when inputs are guaranteed to come from a single layer."""
    gather = build_optimal_single_layer_gather_module_unwrapped(ordinals)
    return LayerInputPipe(input_layer, gather)


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
            {str(l): build_optimal_single_layer_gather_module_unwrapped(per_layer_ordinals[l]) for l in self.layers}
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

        self.final_gather = build_optimal_single_layer_gather_module_unwrapped(concatenated_ordinals)

    def forward(self, layer_values: dict[int, torch.Tensor]):
        with torch.profiler.record_function('GATHER_MULTI_RUN_INDIVIDUAL_LAYERS'):
            layer_inputs_needed = [self.layer_gathers[str(layer)](layer_values[layer]) for layer in self.layers]
        with torch.profiler.record_function('GATHER_MULTI_BROADCAST_SHAPES'):
            layer_shape_hull = torch.broadcast_shapes(*(t.shape[1:] for t in layer_inputs_needed))
        with torch.profiler.record_function('GATHER_MULTI_EXPAND_CONCAT'):
            layer_inputs_needed = torch.concatenate([t.expand(-1, *layer_shape_hull) for t in layer_inputs_needed])

        with torch.profiler.record_function('GATHER_MULTI_RUN_FINAL'):
            out = self.final_gather(layer_inputs_needed)
        return out


def build_optimal_gather_module(input_layer_ordinal_pairs: list[LayerOrdinal]):
    layer0, _ = input_layer_ordinal_pairs[0]
    is_single_layer = all((layer0 == l for l, _ in input_layer_ordinal_pairs[1:]))

    if is_single_layer:
        return build_optimal_single_layer_gather_module(layer0, [o for _, o in input_layer_ordinal_pairs])

    return MultiLayerGather(input_layer_ordinal_pairs)
