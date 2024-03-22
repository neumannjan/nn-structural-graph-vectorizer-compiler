from collections import defaultdict
from typing import Mapping, Protocol, Sequence, overload

import numpy as np
import torch
from torch.jit import unused

from lib.sources.base import LayerOrdinal
from lib.nn.utils import Sequential, ShapeTransformable, SingleLayerOperation, broadcast_shapes_compiled
from lib.utils import detect_repeating_K_sequence_in_list, detect_repeating_sequence_in_list


class GatherLike(ShapeTransformable, Protocol):
    @overload
    def compute_optimal_shape(self) -> list[int]: ...
    @overload
    def compute_optimal_shape(self, in_shape: list[int]) -> list[int]: ...
    @overload
    def compute_optimal_shape(self, layer_shapes: dict[str, list[int]]) -> list[int]: ...

    @property
    def is_optimal(self) -> bool: ...


class GatherModuleLike(GatherLike, Protocol):
    def get_optimal(self) -> "GatherModuleLike": ...

    @overload
    def unwrap_final_gather(self) -> tuple[ShapeTransformable, dict[int, int]]: ...
    @overload
    def unwrap_final_gather(self, shape: list[int]) -> tuple[ShapeTransformable, dict[int, int]]: ...
    @overload
    def unwrap_final_gather(self, layer_shapes: dict[str, list[int]]) -> tuple[ShapeTransformable, dict[int, int]]: ...

    @overload
    def __call__(self) -> torch.Tensor: ...

    @overload
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...

    @overload
    def __call__(self, layer_values: dict[str, torch.Tensor]) -> torch.Tensor: ...


class NoopGather(torch.nn.Module, GatherModuleLike):
    @unused
    def compute_output_shape(self, in_shape: list[int]) -> list[int]:
        return in_shape

    @unused
    def compute_optimal_shape(self, in_shape: list[int]) -> list[int]:
        return self.compute_output_shape(in_shape)

    @unused
    @property
    def is_optimal(self) -> bool:
        return True

    @unused
    def get_optimal(self) -> "NoopGather":
        return self

    @unused
    def unwrap_final_gather(self, shape) -> tuple[ShapeTransformable, dict[int, int]]:
        return self, {}

    def forward(self, x) -> torch.Tensor:
        return x


class TakeValue(torch.nn.Module, GatherModuleLike):
    def __init__(self, ordinal: int) -> None:
        super().__init__()
        self.ordinal = ordinal

    def extra_repr(self) -> str:
        return f"ordinal={self.ordinal}"

    @unused
    def compute_output_shape(self, in_shape: list[int]) -> list[int]:
        return [1, *in_shape[1:]]

    @unused
    def compute_optimal_shape(self, in_shape: list[int]) -> list[int]:
        return self.compute_output_shape(in_shape)

    @unused
    @property
    def is_optimal(self) -> bool:
        return True

    @unused
    def get_optimal(self) -> "TakeValue":
        return self

    @unused
    def unwrap_final_gather(self, shape) -> tuple[ShapeTransformable, dict[int, int]]:
        if self.ordinal == 0:
            return NoopGather(), {}

        return NoopGather(), {0: self.ordinal}

    def forward(self, layer_input: torch.Tensor):
        return layer_input[self.ordinal].unsqueeze(0)


class SliceValues(torch.nn.Module, GatherModuleLike):
    def __init__(self, start: int, end: int) -> None:
        super().__init__()
        self.start = start
        self.end = end

    def extra_repr(self) -> str:
        return f"start={self.start}, end={self.end}"

    @unused
    def compute_output_shape(self, in_shape: list[int]) -> list[int]:
        return [self.end - self.start, *in_shape[1:]]

    @unused
    def compute_optimal_shape(self, in_shape: list[int]) -> list[int]:
        return self.compute_output_shape(in_shape)

    @unused
    @property
    def is_optimal(self) -> bool:
        return True

    @unused
    def get_optimal(self) -> "SliceValues":
        return self

    @unused
    def unwrap_final_gather(self, shape) -> tuple[ShapeTransformable, dict[int, int]]:
        ord_map = {i: j for i, j in enumerate(range(self.start, self.end)) if i != j}
        return NoopGather(), ord_map

    def forward(self, layer_input: torch.Tensor):
        return layer_input[self.start : self.end]


class TakeEachNth(torch.nn.Module, GatherModuleLike):
    def __init__(self, step: int, start: int, end: int) -> None:
        super().__init__()
        self.step = step
        self.start = start
        self.end = end

    def extra_repr(self) -> str:
        return f"step={self.step}, start={self.start}, end={self.end}"

    @unused
    def compute_output_shape(self, in_shape: list[int]) -> list[int]:
        len = self.end - self.start
        len = -(-len // self.step)
        return [len, *in_shape[1:]]

    @unused
    def compute_optimal_shape(self, in_shape: list[int]) -> list[int]:
        return self.compute_output_shape(in_shape)

    @unused
    @property
    def is_optimal(self) -> bool:
        return True

    @unused
    def get_optimal(self) -> "TakeEachNth":
        return self

    @unused
    def unwrap_final_gather(self, shape) -> tuple[ShapeTransformable, dict[int, int]]:
        ord_map = {i: j for i, j in enumerate(range(self.start, self.end, self.step)) if i != j}
        return NoopGather(), ord_map

    def forward(self, layer_input: torch.Tensor):
        return layer_input[self.start : self.end : self.step]


class GenericGather(torch.nn.Module, GatherModuleLike):
    def __init__(self, ordinals: Sequence[int]) -> None:
        super().__init__()
        self.ordinals = torch.nn.Parameter(torch.tensor(ordinals, dtype=torch.int32), requires_grad=False)

    def extra_repr(self) -> str:
        return f"ordinals=(list of size {self.ordinals.shape[0]})"

    @unused
    def compute_output_shape(self, in_shape: list[int]) -> list[int]:
        return [len(self.ordinals), *in_shape[1:]]

    @unused
    def compute_optimal_shape(self, in_shape: list[int]) -> list[int]:
        return self.compute_output_shape(in_shape)

    @unused
    @property
    def is_optimal(self) -> bool:
        return True

    @unused
    def get_optimal(self) -> "GatherModuleLike":
        return self

    @unused
    def unwrap_final_gather(self, shape) -> tuple[ShapeTransformable, dict[int, int]]:
        ord_map = {i: j for i, j in enumerate(self.ordinals.cpu().tolist()) if i != j}
        return NoopGather(), ord_map

    def forward(self, layer_input: torch.Tensor):
        return torch.index_select(layer_input, 0, self.ordinals)


class Repeat(torch.nn.Module, GatherModuleLike):
    def __init__(self, repeats: int, total_length: int) -> None:
        super().__init__()
        self.repeats = repeats
        self.total_length = total_length

    def extra_repr(self) -> str:
        return f"repeats={self.repeats}, total_length={self.total_length}"

    @unused
    def compute_output_shape(self, in_shape: list[int]) -> list[int]:
        return [self.total_length, *in_shape[1:]]

    @unused
    def compute_optimal_shape(self, in_shape: list[int]) -> list[int]:
        return in_shape

    @unused
    @property
    def is_optimal(self) -> bool:
        return False

    @unused
    def get_optimal(self) -> "GatherModuleLike":
        return NoopGather()

    @unused
    def unwrap_final_gather(self, shape) -> tuple[ShapeTransformable, dict[int, int]]:
        ord_map = {}
        i = 0
        for _ in range(self.repeats):
            for j in range(shape[0]):
                if i != j:
                    ord_map[i] = j
                i += 1
                if i >= self.total_length:
                    return NoopGather(), ord_map
        return NoopGather(), ord_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat(self.repeats, *([1] * (x.dim() - 1)))
        x = x[: self.total_length]
        return x


@unused
def unwrap_gathers_sequential(
    shape_like,
    *modules: GatherModuleLike,
) -> tuple[ShapeTransformable, dict[int, int]]:
    out_modules, out_idx_map = [], {}

    for module in modules:
        tpl = module.unwrap_final_gather(shape_like)
        if tpl is None:
            continue

        module, idx_map = tpl

        # combine out_modules and module
        out_modules.append(module)

        # combine out_idx_map and idx_map
        for a, b in out_idx_map.items():
            c = idx_map.get(b, b)
            if a == c:
                if a in out_idx_map:
                    del out_idx_map[a]
            else:
                out_idx_map[a] = c

    if len(out_modules) == 0:
        return NoopGather(), out_idx_map

    out_modules = [m for m in out_modules if not isinstance(m, NoopGather)]

    if len(out_modules) == 0:
        return NoopGather(), out_idx_map
    elif len(out_modules) == 1:
        return out_modules[0], out_idx_map
    else:
        return Sequential(out_modules), out_idx_map


class GatherAndRepeat(torch.nn.Module, GatherModuleLike):
    def __init__(self, the_gather_module: GatherModuleLike, repeats: int, total_length: int) -> None:
        super().__init__()
        self.gather = the_gather_module
        # Repeat is slow !!
        self.repeat = Repeat(repeats, total_length)

    @unused
    def compute_output_shape(self, shape_like) -> list[int]:
        shape_like = self.gather.compute_output_shape(shape_like)
        shape_like = self.repeat.compute_output_shape(shape_like)
        return shape_like

    @unused
    def compute_optimal_shape(self, in_shape: list[int]) -> list[int]:
        return self.gather.compute_optimal_shape(in_shape)

    @unused
    @property
    def is_optimal(self) -> bool:
        return False

    @unused
    def get_optimal(self) -> "GatherModuleLike":
        return self.gather.get_optimal()

    @unused
    def unwrap_final_gather(self, shape_like) -> tuple[ShapeTransformable, dict[int, int]]:
        return unwrap_gathers_sequential(shape_like, self.gather, self.repeat)

    def forward(self, x) -> torch.Tensor:
        x = self.gather(x)
        x = self.repeat(x)
        return x


def build_optimal_gather(
    ordinals: Sequence[int],
    allow_subseq=True,
    period_hint: int | None = None,
) -> TakeValue | SliceValues | TakeEachNth | GenericGather | GatherAndRepeat:
    ###### simple retrieval ######

    all_inputs_the_same = all((ordinals[0] == o for o in ordinals[1:]))

    if all_inputs_the_same:
        return TakeValue(ordinals[0])

    ###### simple slicing #######

    step = ordinals[1] - ordinals[0]
    all_ordinals_differ_by_step = all((b - a == step for a, b in zip(ordinals[:-1], ordinals[1:])))

    if all_ordinals_differ_by_step:
        if step == 1:
            return SliceValues(ordinals[0], ordinals[-1] + 1)

        return TakeEachNth(step=step, start=ordinals[0], end=ordinals[-1] + 1)

    ###### subsequence with (optimizable) repeat: #######

    if allow_subseq:
        subseq = None
        if period_hint is not None:
            # try the hint first
            subseq = detect_repeating_K_sequence_in_list(ordinals, period=period_hint, allow_last_incomplete=True)

        if subseq is None:
            subseq = detect_repeating_sequence_in_list(ordinals, allow_last_incomplete=True)

        if subseq is not None and len(subseq) <= len(ordinals) // 2:
            subseq_gather = build_optimal_gather(subseq.tolist(), allow_subseq=False)
            repeats = -(-len(ordinals) // len(subseq))
            total_length = len(ordinals)

            if total_length == len(subseq):
                return subseq_gather

            return GatherAndRepeat(subseq_gather, repeats=repeats, total_length=total_length)

    ###### generic fallback implementation ######

    return GenericGather(ordinals)


class SingleLayerGather(SingleLayerOperation, GatherModuleLike):
    def __init__(self, input_layer: int, the_gather: GatherModuleLike) -> None:
        super().__init__(input_layer, the_gather)
        self.delegate_gather = the_gather

    @unused
    def compute_optimal_shape(self, layer_shapes: dict[str, list[int]]) -> list[int]:
        return self.delegate_gather.compute_optimal_shape(layer_shapes[str(self.input_layer)])

    @unused
    @property
    def is_optimal(self) -> bool:
        return self.delegate_gather.is_optimal

    @unused
    def get_optimal(self) -> "SingleLayerGather":
        if self.is_optimal:
            return self

        optimal_delegate = self.delegate_gather.get_optimal()
        return SingleLayerGather(input_layer=self.input_layer, the_gather=optimal_delegate)

    @unused
    def unwrap_final_gather(self, shape_like) -> tuple[ShapeTransformable, dict[int, int]]:
        mdl, idx_map = self.delegate_gather.unwrap_final_gather(shape_like)

        if mdl == self.delegate:
            return self, idx_map

        return SingleLayerOperation(self.input_layer, mdl), idx_map


def build_optimal_single_layer_gather(input_layer: int, ordinals: list[int], period_hint: int | None = None):
    """Build the optimal gather network module when inputs are guaranteed to come from a single layer."""
    gather = build_optimal_gather(ordinals, period_hint=period_hint)
    return SingleLayerGather(input_layer, gather)


def _expand_concat_tensors(xs: list[torch.Tensor]) -> torch.Tensor:
    # TODO: Do we really need broadcasting here? Can we resolve this in a simpler way
    # (such as ideally by knowing the input dimensions)?
    shapes = [list(t.shape[1:]) for t in xs]

    shape_hull = [-1]
    shape_hull.extend(broadcast_shapes_compiled(shapes))

    xs = [t.expand(shape_hull) for t in xs]

    x = torch.concatenate(xs)
    return x


def _compute_expand_concat_shape(shapes: list[list[int]]) -> list[int]:
    first_dim = sum((s[0] for s in shapes))
    rest_dims = torch.broadcast_shapes(*[s[1:] for s in shapes])
    return [first_dim, *rest_dims]


class ConcatLayers(torch.nn.Module, GatherModuleLike):
    """Concatenate inputs from multiple layers."""

    def __init__(self, layers: list[int], layer_offsets: list[int]) -> None:
        super().__init__()
        self.layers = [str(l) for l in layers]
        self._layer_id_to_offset_map: dict[int, int] = {id: off for id, off in zip(layers, layer_offsets)}

    @unused
    def get_layer_ordinal_position(self, o: LayerOrdinal) -> int:
        return self._layer_id_to_offset_map[o.layer] + o.ordinal

    @unused
    def compute_optimal_shape(self, layer_shapes: dict[str, list[int]]) -> list[int]:
        return self.compute_output_shape(layer_shapes)

    @unused
    @property
    def is_optimal(self) -> bool:
        return True

    @unused
    def get_optimal(self) -> "ConcatLayers":
        return self

    @unused
    def unwrap_final_gather(self, shape) -> tuple[ShapeTransformable, dict[int, int]]:
        # cannot unwrap a concat of multiple layers! :(
        return self, {}

    @unused
    def compute_output_shape(self, layer_shapes: dict[str, list[int]]) -> list[int]:
        return _compute_expand_concat_shape([layer_shapes[l] for l in self.layers])

    def forward(self, layer_values: dict[str, torch.Tensor]) -> torch.Tensor:
        xs = [layer_values[l] for l in self.layers]
        x = _expand_concat_tensors(xs)
        return x

    def extra_repr(self) -> str:
        return "[" + ", ".join(self.layers) + "]"


class GatherConcatLayers(torch.nn.Module, GatherModuleLike):
    """
    Gather for inputs coming from multiple layers.

    First performs individual gathers for each input layer. Then concatenates the result to a single tensor.
    Provides a mapping to remember which value contains which neuron output.
    """

    def __init__(self, per_layer_ordinals: dict[int, list[int]]) -> None:
        super().__init__()
        self.layers = sorted(per_layer_ordinals.keys(), reverse=True)
        layers_map = {l: i for i, l in enumerate(self.layers)}

        ### setup single-layer gathers ###

        # each gather is a set gather, so there is no optimal period for any of them
        self.layer_gathers = torch.nn.ModuleDict(
            {str(l): build_optimal_gather(per_layer_ordinals[l]) for l in self.layers}
        )

        ### setup idx map ###

        layer_sizes = [len(per_layer_ordinals[layer]) for layer in self.layers]
        layer_prefixes = np.concatenate([[0], np.cumsum(layer_sizes)[:-1]])

        self._layer_ordinal_positions: dict[LayerOrdinal, int] = {
            LayerOrdinal(l, o): layer_prefixes[layers_map[l]] + i
            for l in self.layers
            for i, o in enumerate(per_layer_ordinals[l])
        }

    @unused
    def get_layer_ordinal_position(self, o: LayerOrdinal) -> int:
        return self._layer_ordinal_positions[o]

    def __getitem__(self, layer_id: int):
        return self.layer_gathers[str(layer_id)]

    @unused
    def compute_output_shape(self, layer_shapes: dict[str, list[int]]) -> list[int]:
        layer_output_shapes: list[list[int]] = [
            self.layer_gathers[str(l)].compute_output_shape(layer_shapes[str(l)]) for l in self.layers
        ]
        return _compute_expand_concat_shape(layer_output_shapes)

    @unused
    def compute_optimal_shape(self, layer_shapes: dict[str, list[int]]) -> list[int]:
        return self.compute_output_shape(layer_shapes)

    @unused
    @property
    def is_optimal(self) -> bool:
        return True

    @unused
    def get_optimal(self) -> "GatherConcatLayers":
        return self

    @unused
    def unwrap_final_gather(self, shape) -> tuple[ShapeTransformable, dict[int, int]]:
        # cannot unwrap a concat of multiple layers! :(
        return self, {}

    def forward(self, layer_values: dict[str, torch.Tensor]):
        xs = []

        for str_layer, gather in self.layer_gathers.items():
            x_this = gather(layer_values[str_layer])
            xs.append(x_this)

        x = _expand_concat_tensors(xs)
        return x

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}({repr(self.layer_gathers)})"


class MultiLayerGather(torch.nn.Module, GatherModuleLike):
    """
    Gather for inputs coming from multiple layers.

    First, concatenates the outputs of all input layers into a single tensor. This can be done either
    with a simple concat operation (a `ConcatLayers`), or with a unique-concat (`GatherConcatLayers`),
    where only the needed values are gathered from each individual layer first before concatenating.

    Finally, performs the non-set gather using a final single Gather.
    """

    def __init__(self, multi_layer_concat: ConcatLayers | GatherConcatLayers, final_gather: GatherModuleLike) -> None:
        super().__init__()
        self.multi_layer_concat = multi_layer_concat
        self.final_gather = final_gather

    @unused
    def compute_output_shape(self, shape_like) -> list[int]:
        shape_like = self.multi_layer_concat.compute_output_shape(shape_like)
        shape_like = self.final_gather.compute_output_shape(shape_like)
        return shape_like

    @unused
    def compute_optimal_shape(self, shape_like) -> list[int]:
        return self.compute_output_shape(shape_like)

    @unused
    @property
    def is_optimal(self) -> bool:
        return self.final_gather.is_optimal

    @unused
    def get_optimal(self) -> "MultiLayerGather":
        if self.is_optimal:
            return self

        final_optimal = self.final_gather.get_optimal()
        return MultiLayerGather(self.multi_layer_concat, final_optimal)

    @unused
    def unwrap_final_gather(self, shape) -> tuple[ShapeTransformable, dict[int, int]]:
        mdl, idx_map = self.final_gather.unwrap_final_gather()
        if isinstance(mdl, NoopGather):
            return self.multi_layer_concat, idx_map

        if mdl == self.final_gather:
            return self, idx_map

        return Sequential([self.multi_layer_concat, mdl]), idx_map

    def forward(self, layer_values: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.multi_layer_concat(layer_values)
        x = self.final_gather(x)
        return x


def _build_multi_layer_gather(
    inputs_ordinals: list[LayerOrdinal],
    layer_shapes: Mapping[str, list[int]],
    use_unique_pre_gathers: bool,
    period_hint: int | None = None,
):
    if use_unique_pre_gathers:
        per_layer_ordinals_set: dict[int, set[int]] = defaultdict(lambda: set())

        for l, o in inputs_ordinals:
            per_layer_ordinals_set[l].add(o)

        per_layer_ordinals = {l: sorted(o) for l, o in per_layer_ordinals_set.items()}

        multi_layer_concat = GatherConcatLayers(per_layer_ordinals)
    else:
        # TODO reverse=True is probably better, but it should work either way
        layers_to_gather = sorted({o.layer for o in inputs_ordinals}, reverse=False)
        layer_sizes_list = np.array([layer_shapes[str(l)][0] for l in layers_to_gather[:-1]])
        layer_offsets = np.concatenate([[0], np.cumsum(layer_sizes_list)]).tolist()
        multi_layer_concat = ConcatLayers(layers_to_gather, layer_offsets)

    final_ordinals = [multi_layer_concat.get_layer_ordinal_position(o) for o in inputs_ordinals]
    final_gather = build_optimal_gather(final_ordinals, period_hint=period_hint)

    return MultiLayerGather(multi_layer_concat, final_gather)


def build_optimal_multi_layer_gather(
    inputs_ordinals: list[LayerOrdinal],
    layer_shapes: Mapping[str, list[int]],
    use_unique_pre_gathers: bool,
    period_hint: int | None = None,
):
    layer0, _ = inputs_ordinals[0]
    is_single_layer = all((layer0 == l for l, _ in inputs_ordinals[1:]))

    if is_single_layer:
        return build_optimal_single_layer_gather(layer0, [o for _, o in inputs_ordinals], period_hint=period_hint)

    return _build_multi_layer_gather(
        inputs_ordinals,
        layer_shapes,
        use_unique_pre_gathers=use_unique_pre_gathers,
        period_hint=period_hint,
    )


class ViewWithPeriod(torch.nn.Module, GatherModuleLike):
    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = period

    def extra_repr(self) -> str:
        return f"period={self.period}"

    @unused
    def compute_output_shape(self, shape_like) -> list[int]:
        l = shape_like[0]
        return [l // self.period, self.period, *shape_like[1:]]

    @unused
    def compute_optimal_shape(self, shape_like) -> list[int]:
        return self.compute_output_shape(shape_like)

    @unused
    @property
    def is_optimal(self) -> bool:
        return True

    @unused
    def get_optimal(self) -> "GatherModuleLike":
        return self

    @unused
    def unwrap_final_gather(self, shape) -> tuple[ShapeTransformable, dict[int, int]]:
        return self, {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = [-1, self.period]
        shape.extend(x.shape[1:])
        return x.view(shape)


class GatherAndView(torch.nn.Module, GatherModuleLike):
    def __init__(self, gather: GatherModuleLike, view: ViewWithPeriod) -> None:
        super().__init__()
        self.gather = gather
        self.view = view

    @unused
    def compute_output_shape(self, shape_like) -> list[int]:
        shape_like = self.gather.compute_output_shape(shape_like)
        shape_like = self.view.compute_output_shape(shape_like)
        return shape_like

    @unused
    def compute_optimal_shape(self, shape_like) -> list[int]:
        shape_like = self.gather.compute_optimal_shape(shape_like)
        shape_like = self.view.compute_output_shape(shape_like)
        return shape_like

    @unused
    @property
    def is_optimal(self) -> bool:
        return self.gather.is_optimal

    @unused
    def get_optimal(self) -> "GatherAndView | ViewWithPeriod":
        if self.is_optimal:
            return self

        gather_optimal = self.gather.get_optimal()

        if isinstance(gather_optimal, NoopGather):
            return self.view

        return GatherAndView(gather=gather_optimal, view=self.view)

    @unused
    def unwrap_final_gather(self, shape) -> tuple[ShapeTransformable, dict[int, int]]:
        return self, {}

    def forward(self, x) -> torch.Tensor:
        x = self.gather(x)
        x = self.view(x)
        return x


def build_optimal_gather_and_reshape(ordinals: list[int], period: int | None):
    gather = build_optimal_gather(ordinals, period_hint=period)

    if period is None:
        return gather

    reshape = ViewWithPeriod(period=period)
    return GatherAndView(gather, reshape)


def build_optimal_multi_layer_gather_and_reshape(
    inputs_ordinals: list[LayerOrdinal],
    layer_shapes: Mapping[str, list[int]],
    use_unique_pre_gathers: bool,
    period: int | None,
):
    gather = build_optimal_multi_layer_gather(
        inputs_ordinals,
        layer_shapes,
        use_unique_pre_gathers=use_unique_pre_gathers,
        period_hint=period,
    )

    if period is None:
        return gather

    reshape = ViewWithPeriod(period=period)
    return GatherAndView(gather, reshape)


def get_optimal_gather_for_periodic_gather(gather: GatherModuleLike, period: int, input_shape_like) -> GatherModuleLike:
    if not gather.is_optimal:
        if input_shape_like is None:
            optimal_period = gather.compute_optimal_shape()[1]
        else:
            optimal_period = gather.compute_optimal_shape(input_shape_like)[1]

        if optimal_period == period:
            return gather if gather.is_optimal else gather.get_optimal()
        elif period % optimal_period == 0:
            raise NotImplementedError()

    return gather
