import itertools
from typing import Collection, Iterable, Protocol, Sequence

import numpy as np
import torch

from lib.nn.gather import (
    GatherAndRepeatNonOptimal,
    GatherAndReshape,
    NoopGather,
    build_optimal_gather,
    build_optimal_gather_and_reshape,
)
from lib.nn.sources.source import WeightDefinition
from lib.nn.utils.utils import ViewWithPeriod
from lib.utils import detect_repeating_sequence_in_list


class WeightLike(Protocol):
    def apply_to(self, to: torch.Tensor) -> torch.Tensor:
        ...

    def expand_to(self, shape: torch.Size | tuple[int, ...] | list[int], shape_single: bool) -> torch.Tensor:
        ...

    def expand_as_weight(self, shape: torch.Size | tuple[int, ...] | list[int], shape_single: bool) -> "WeightLike":
        ...

    @property
    def is_multiple(self) -> bool:
        ...

    @property
    def value(self) -> torch.nn.Parameter:
        ...

    @property
    def shape(self) -> torch.Size:
        ...

    @property
    def learnable(self) -> bool:
        ...

    def as_parameter(self) -> torch.nn.Parameter:
        ...

    def unsqueeze0(self) -> "WeightLike":
        ...


class WeightModuleLike(Protocol):
    @property
    def shape(self) -> torch.Size:
        ...

    @property
    def is_multiple(self) -> bool:
        ...

    def __call__(self) -> torch.Tensor:
        ...


def _get_single_shape(shape: torch.Size | tuple[int, ...] | list[int], shape_single: bool):
    if shape_single:
        return shape
    else:
        return shape[1:]


def get_weight_shape_single(weight: WeightLike | WeightModuleLike):
    if weight.is_multiple:
        return weight.shape[1:]
    else:
        return weight.shape


class Weight(torch.nn.Module, WeightLike, WeightModuleLike):
    def __init__(self, weight: torch.nn.Parameter, learnable: bool) -> None:
        super().__init__()
        self.weight = weight
        self._learnable = learnable

        if weight.requires_grad != learnable:
            lrn = "learnable" if learnable else "not learnable"

            raise ValueError(f"For weight that is {lrn}, it must hold that requires_grad == {learnable}")

    def apply_to(self, to: torch.Tensor) -> torch.Tensor:
        return self.weight @ to

    @property
    def value(self) -> torch.nn.Parameter:
        return self.weight

    @property
    def is_multiple(self) -> bool:
        return False

    def expand_to(self, shape: torch.Size | tuple[int, ...] | list[int], shape_single: bool) -> torch.Tensor:
        shape = _get_single_shape(shape, shape_single)

        if self.learnable and sum(shape) != sum(self.weight.shape):
            raise ValueError("Cannot expand learnable weight unless the total no. of scalar params is the same!")

        return self.weight.view(shape)

    def expand_as_weight(self, shape: torch.Size | tuple[int, ...] | list[int], shape_single: bool) -> "Weight":
        tensor = self.expand_to(shape, shape_single)
        return Weight(torch.nn.Parameter(tensor, requires_grad=self.learnable), learnable=self.learnable)

    def extra_repr(self) -> str:
        shape = "x".join(map(str, self.weight.shape))
        return f"{shape}, learnable={self.learnable}"

    def forward(self) -> torch.Tensor:
        return self.weight

    @property
    def shape(self) -> torch.Size:
        return self.weight.shape

    @property
    def learnable(self) -> bool:
        return self._learnable

    def as_parameter(self) -> torch.nn.Parameter:
        return self.weight

    def unsqueeze0(self) -> "Weights":
        return Weights(
            torch.nn.Parameter(self.weight.unsqueeze(0), requires_grad=self.learnable), learnable=self.learnable
        )


class UnitWeight(Weight, WeightLike, WeightModuleLike):
    def __init__(self, tensor: torch.Tensor | None = None) -> None:
        if tensor is None:
            tensor = torch.tensor([1.0])

        super().__init__(torch.nn.Parameter(tensor, requires_grad=False), learnable=False)

    @property
    def is_multiple(self) -> bool:
        return False

    def expand_to(self, shape: torch.Size | tuple[int, ...] | list[int], shape_single: bool) -> torch.Tensor:
        shape = _get_single_shape(shape, shape_single)

        if len(shape) == 2 and shape[0] == shape[1]:
            return torch.eye(shape[0])

        return torch.ones(shape)

    def expand_as_weight(self, shape: torch.Size | tuple[int, ...] | list[int], shape_single: bool) -> "UnitWeight":
        tensor = self.expand_to(shape, shape_single)
        return UnitWeight(tensor)

    def apply_to(self, to: torch.Tensor) -> torch.Tensor:
        return to

    @property
    def learnable(self) -> bool:
        return False

    def unsqueeze0(self) -> "UnitWeights":
        return UnitWeights(self.weight.unsqueeze(0))


class UnitWeights(Weight, WeightLike, WeightModuleLike):
    def __init__(self, tensor: torch.Tensor | None = None) -> None:
        if tensor is None:
            tensor = torch.tensor([1.0])

        super().__init__(torch.nn.Parameter(tensor, requires_grad=False), learnable=False)

    @property
    def is_multiple(self) -> bool:
        return True

    def expand_to(self, shape: torch.Size | tuple[int, ...] | list[int], shape_single: bool) -> torch.Tensor:
        shape = _get_single_shape(shape, shape_single)

        if len(shape) == 2 and shape[0] == shape[1]:
            out = torch.eye(shape[0]).unsqueeze(0)
        else:
            out = torch.ones(shape).unsqueeze(0)

        out_shape = list(out.shape)
        out_shape[0] = self.shape[0]

        out = out.view(out_shape)
        return out

    def expand_as_weight(self, shape: torch.Size | tuple[int, ...] | list[int], shape_single: bool) -> "UnitWeights":
        tensor = self.expand_to(shape, shape_single)
        return UnitWeights(tensor)

    def apply_to(self, to: torch.Tensor) -> torch.Tensor:
        return to

    @property
    def learnable(self) -> bool:
        return False

    def unsqueeze0(self) -> "UnitWeights":
        return self


class Weights(Weight, WeightLike, WeightModuleLike):
    def __init__(self, weights: torch.nn.Parameter, learnable: bool) -> None:
        super().__init__(weights, learnable)

    @property
    def is_multiple(self) -> bool:
        return True

    def expand_to(self, shape: torch.Size | tuple[int, ...] | list[int], shape_single: bool) -> torch.Tensor:
        shape = _get_single_shape(shape, shape_single)

        if self.learnable and sum(shape) != sum(self.weight.shape):
            raise ValueError("Cannot expand learnable weight unless the total no. of scalar params is the same!")

        out_shape = [self.shape[0]]
        out_shape.extend(shape)

        return self.weight.view(shape)

    def expand_as_weight(self, shape: torch.Size | tuple[int, ...] | list[int], shape_single: bool) -> "Weights":
        tensor = self.expand_to(shape, shape_single)

        return Weights(torch.nn.Parameter(tensor, requires_grad=self.learnable), learnable=self.learnable)

    def unsqueeze0(self) -> "Weights":
        return self


def create_weight(tensor: torch.Tensor, is_learnable: bool) -> Weight:
    if not is_learnable and (tensor == torch.tensor([1.0], device=tensor.device)).all():
        return UnitWeight()

    return Weight(torch.nn.Parameter(torch.atleast_1d(tensor), requires_grad=is_learnable), learnable=is_learnable)


class StackWeights(torch.nn.Module, WeightModuleLike):
    def __init__(self, weights_unsqueezed: Sequence[torch.nn.Parameter]) -> None:
        super().__init__()
        self.weights = torch.nn.ParameterList([w for w in weights_unsqueezed])

    def forward(self) -> torch.Tensor:
        return torch.concatenate(tuple(self.weights))

    @property
    def shape(self) -> torch.Size:
        n_weights = sum((w.shape[0] for w in self.weights))

        shape = [n_weights]
        shape.extend(self.weights[0].shape)

        return torch.Size(shape)

    @property
    def is_multiple(self) -> bool:
        return True


class ExpandWeight(torch.nn.Module, WeightModuleLike):
    def __init__(self, weight: Weight, shape: torch.Size | tuple[int, ...] | list[int], shape_single: bool) -> None:
        super().__init__()
        self.weight = weight
        self._shape = shape if shape_single else shape[1:]

    def forward(self) -> torch.Tensor:
        return self.weight.expand_to(self._shape, shape_single=True)

    @property
    def shape(self) -> torch.Size:
        shape = []

        if self.is_multiple:
            shape.append(self.weight.shape[0])

        shape.extend(self._shape)

        return torch.Size(shape)

    @property
    def is_multiple(self) -> bool:
        return self.weight.is_multiple


class StackWeightModules(torch.nn.Module, WeightModuleLike):
    def __init__(self, modules_unsqueezed: Sequence[WeightModuleLike]) -> None:
        super().__init__()

        assert all((m.is_multiple for m in modules_unsqueezed))

        the_modules = []

        for module in modules_unsqueezed:
            if isinstance(module, StackWeightModules):
                the_modules.extend(module.the_modules)
            else:
                the_modules.append(module)

        self.the_modules = torch.nn.ModuleList(the_modules)

    @property
    def weight_modules(self) -> Sequence[WeightModuleLike]:
        return self.the_modules

    @property
    def shape(self) -> torch.Size:
        n_weights = sum((m.shape[0] for m in self.weight_modules))
        shape = [n_weights]
        shape.extend(self.weight_modules[0].shape[1:])

        return torch.Size(shape)

    @property
    def is_multiple(self) -> bool:
        return True

    def forward(self) -> torch.Tensor:
        modules = [m() for m in self.the_modules]
        return torch.concatenate(modules)


def _stack_weights_nonlearnable(
    weights: Sequence[Weight | None], shape_hull: torch.Size | tuple[int, ...] | list[int] | None = None
) -> Weights | None:
    ws = [w for w in weights if w is not None]
    assert all((not w.learnable for w in ws))

    if len(ws) == 0:
        return None

    ws = [w.unsqueeze0() for w in ws]
    shapes = [w.shape[1:] for w in ws]

    if shape_hull is None:
        shape_hull = torch.broadcast_shapes(*shapes)

    ws = [w.expand_as_weight(shape_hull, shape_single=True) if w.shape[1:] != shape_hull else w for w in ws]
    if len(ws) == 1:
        w = ws[0]
    else:
        w = torch.concatenate([w.as_parameter() for w in ws])
        w = Weights(torch.nn.Parameter(w, requires_grad=False), learnable=False)
    return w


def _stack_weights_by_sum_match(
    weights: Sequence[Weight | None],
    shape_hull: torch.Size | tuple[int, ...] | list[int] | None = None,
    group_learnable_weight_parameters=True,
) -> Weights | StackWeightModules | None:
    ws = [w for w in weights if w is not None]

    if len(ws) == 0:
        return None

    ws = [w.unsqueeze0() for w in ws]

    shapes = [w.shape[1:] for w in ws]

    if shape_hull is None:
        shape_hull = torch.broadcast_shapes(*shapes)

    shape_hull_sum = sum(shape_hull)

    ws_final: list[Weights | StackWeightModules] = []

    id_provider = iter(itertools.count())

    def _get_group(w: Weights):
        weight_shape_sum_matches_hull = sum(_get_single_shape(w.shape, shape_single=False)) == shape_hull_sum
        learnable = w.learnable

        if group_learnable_weight_parameters or not learnable:
            id = -1
        else:
            id = next(id_provider)

        return weight_shape_sum_matches_hull, learnable, id

    ws_grouped = itertools.groupby(ws, key=_get_group)

    for (group_shape_sum_matches_hull, learnable, _), group in ws_grouped:
        if group_shape_sum_matches_hull:
            ws_this = [
                w.expand_as_weight(shape_hull, shape_single=True) if w.shape[1:] != shape_hull else w for w in group
            ]

            if len(ws_this) == 1:
                w_this = ws_this[0]
            else:
                w_this = torch.concatenate([w.as_parameter() for w in ws_this])
                w_this = Weights(torch.nn.Parameter(w_this, requires_grad=learnable), learnable=learnable)
        else:
            ws_this = [ExpandWeight(w, shape_hull, shape_single=True) for w in group]
            w_this = StackWeightModules(ws_this)

        ws_final.append(w_this)

    if len(ws_final) == 1:
        w_final = ws_final[0]
    else:
        w_final = StackWeightModules(ws_final)

    return w_final


def _create_weights_set(weights: Iterable[WeightDefinition], out_map: dict[int, int], weights_out: list[Weight]):
    for weight in weights:
        w_idx = weight.id

        if w_idx not in out_map:
            out_map[w_idx] = len(out_map)

            w_tensor = weight.get_value_torch()
            weight = create_weight(w_tensor, is_learnable=weight.learnable)
            weights_out.append(weight)


def create_weights_using_packing_strategy(
    weight_definitions: Collection[WeightDefinition], group_learnable_weight_parameters=True
):
    idx_map: dict[int, int] = {}

    weights_learnable: list[Weight] = []
    weights_nonlearnable: list[Weight] = []

    _create_weights_set(filter(lambda w: w.learnable, weight_definitions), idx_map, weights_learnable)
    _create_weights_set(filter(lambda w: not w.learnable, weight_definitions), idx_map, weights_nonlearnable)

    shapes_all = [_get_single_shape(w.shape, shape_single=not w.is_multiple) for w in weights_learnable]

    shapes_all.extend([_get_single_shape(w.shape, shape_single=not w.is_multiple) for w in weights_nonlearnable])

    shape_hull = torch.broadcast_shapes(*shapes_all)

    w_learnable = _stack_weights_by_sum_match(
        weights_learnable, shape_hull, group_learnable_weight_parameters=group_learnable_weight_parameters
    )
    w_nonlearnable = _stack_weights_nonlearnable(weights_nonlearnable, shape_hull)

    if w_learnable is not None and w_nonlearnable is not None:
        out = StackWeightModules([w_learnable, w_nonlearnable])
    elif w_learnable is not None:
        out = w_learnable
    elif w_nonlearnable is not None:
        out = w_nonlearnable
    else:
        raise ValueError("There are no weights!")

    return out, idx_map


def _check_is_each_learnable_weight_used_only_once(
    weight_definitions: Collection[WeightDefinition], period: int | None
) -> tuple[bool, Collection[WeightDefinition]]:
    weight_definitions_dict = {wd.id: wd for wd in weight_definitions}

    ids_order = np.array([wd.id for wd in weight_definitions])

    weight_definitions_modified = weight_definitions

    if period is not None:
        # TODO: make a `detect_repeating_K_sequence_in_list`
        subseq = detect_repeating_sequence_in_list(ids_order, allow_last_incomplete=False)
        if subseq is not None and len(subseq) == period:
            ids_order = subseq
            weight_definitions_modified = [weight_definitions_dict[id] for id in subseq]

    id_counts = dict(np.stack(np.unique(ids_order, return_counts=True), axis=0).T.tolist())

    out = all((v == 1 for wd_id, v in id_counts.items() if weight_definitions_dict[wd_id].learnable))

    if out:
        return out, weight_definitions_modified
    else:
        return out, weight_definitions


def create_weights_and_gather(
    weight_definitions: Collection[WeightDefinition],
    period: int | None = None,
    group_learnable_weight_parameters=True,
):
    n_orig_weight_definitions = len(weight_definitions)
    each_learnable_used_only_once, weight_definitions = _check_is_each_learnable_weight_used_only_once(
        weight_definitions, period=period
    )

    if not each_learnable_used_only_once:
        # must default to creating the gather and the weight independently

        weight, idx_map = create_weights_using_packing_strategy(
            weight_definitions, group_learnable_weight_parameters=group_learnable_weight_parameters
        )

        weight_idxs: list[int] = [idx_map[wd.id] for wd in weight_definitions]

        if period is None:
            gather = build_optimal_gather(weight_idxs)
        else:
            gather = build_optimal_gather_and_reshape(weight_idxs, period=period)

        return weight, gather

    # can create the weights already in order (such that no gather operation is needed)
    # TODO: write a `detect_repeating_K_sequence_in_list` and make sure that the gather gets the hint
    weights: list[Weight] = [
        create_weight(wd.get_value_torch(), is_learnable=wd.learnable) for wd in weight_definitions
    ]

    shapes_all = [_get_single_shape(w.shape, shape_single=not w.is_multiple) for w in weights]
    shape_hull = torch.broadcast_shapes(*shapes_all)

    # expand all nonlearnable weights
    weights = [w if w.learnable else w.expand_as_weight(shape_hull, shape_single=True) for w in weights]
    weight = _stack_weights_by_sum_match(
        weights, shape_hull=shape_hull, group_learnable_weight_parameters=group_learnable_weight_parameters
    )
    assert weight is not None

    gather = NoopGather(n_items=len(weight_definitions))

    if n_orig_weight_definitions > len(weight_definitions):
        gather = GatherAndRepeatNonOptimal(
            gather,
            repeats=-(-n_orig_weight_definitions // len(weight_definitions)),
            total_length=n_orig_weight_definitions,
        )

    if period is not None:
        gather = GatherAndReshape(gather, ViewWithPeriod(period))
    return weight, gather
