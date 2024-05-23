from collections.abc import Sequence
from typing import Literal

from lib.vectorize.model.fact import Fact, fact_repr
from lib.vectorize.model.gather import Gather
from lib.vectorize.model.reduce import Reduce
from lib.vectorize.model.refs import LayerRefs, Refs
from lib.vectorize.model.repr import my_repr, repr_module_like, repr_slots
from lib.vectorize.model.shape import Shape, VariousShape
from lib.vectorize.model.transform import Transform


class GatheredLayers:
    __slots__ = ("refs", "gather")
    __repr__ = repr_slots

    def __init__(self, refs: LayerRefs, gather: Gather) -> None:
        self.refs = refs
        self.gather = gather

    def __hash__(self) -> int:
        return hash((self.refs, self.gather))

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, GatheredLayers) and self.refs == value.refs and self.gather == value.gather


Input = GatheredLayers | Refs


class InputLayerBase:
    __slots__ = ("input",)
    __repr__ = repr_slots

    def __init__(self, input: Input) -> None:
        self.input = input

    def __hash__(self) -> int:
        return hash(self.input)

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, InputLayerBase) and self.input == value.input


DimensionLift = tuple[Literal[-1], int] | tuple[int, Literal[-1]]
DimensionLifts = tuple[DimensionLift, DimensionLift] | None


def _get_lift_dimension(lift: DimensionLift) -> Literal[0, 1]:
    if lift[0] == -1 and lift[1] == -1:
        raise ValueError(lift)
    elif lift[1] == -1:
        return 0
    elif lift[0] == -1:
        return 1
    else:
        raise ValueError(lift)

def lifts_dimension_match(lifts: DimensionLifts) -> bool:
    if lifts is None:
        raise ValueError()

    a, b = lifts
    return _get_lift_dimension(a) == _get_lift_dimension(b)


def get_lifts_period(lifts: DimensionLifts) -> int | None:
    if lifts is None:
        return None

    la, lb = lifts

    if la[0] == lb[0] == -1 and la[1] == lb[1]:
        return la[1]

    if la[1] == lb[1] == -1 and la[0] == lb[0]:
        return la[0]

    return None


class LinearLayerBase:
    __slots__ = ("input", "weight", "lifts")
    __repr__ = repr_slots

    def __init__(
        self,
        input: Input,
        weight: Input,
        lifts: DimensionLifts = None,
    ) -> None:
        self.input = input
        self.weight = weight
        self.lifts = lifts

    def __hash__(self) -> int:
        return hash((self.input, self.weight, self.lifts))

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, LinearLayerBase)
            and self.input == value.input
            and self.weight == value.weight
            and self.lifts == value.lifts
        )

    @property
    def lifts_dimension_match(self) -> bool:
        return lifts_dimension_match(self.lifts)


class LinearGatherLayerBase:
    __slots__ = ("input", "weight", "gather", "lifts")
    __repr__ = repr_slots

    def __init__(
        self,
        input: Input,
        weight: Input,
        gather: Gather,
        lifts: DimensionLifts = None,
    ) -> None:
        self.input = input
        self.weight = weight
        self.gather = gather
        self.lifts = lifts

    def __hash__(self) -> int:
        return hash((self.input, self.weight, self.gather, self.lifts))

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, LinearGatherLayerBase)
            and self.input == value.input
            and self.weight == value.weight
            and self.gather == value.gather
            and self.lifts == value.lifts
        )

    @property
    def lifts_dimension_match(self) -> bool:
        return lifts_dimension_match(self.lifts)


LayerBase = InputLayerBase | LinearLayerBase | LinearGatherLayerBase


_LAYER_MODULE_LIKES = ("base", "aggregate", "transform")


class Layer:
    __slots__ = ("base", "aggregate", "transform", "count", "shape", "ord_map", "compilable")
    __repr__ = repr_slots

    def __init__(
        self,
        base: LayerBase,
        aggregate: Reduce,
        transform: Transform,
        count: int | None = None,
        shape: Shape | None = None,
        compilable: bool = False,
    ) -> None:
        self.base = base
        self.aggregate = aggregate
        self.transform = transform
        self.count = count
        self.shape = shape if shape is not None else VariousShape()
        self.ord_map: dict[int, int] = {}
        self.compilable = compilable

    def __repr__(self) -> str:
        out = self.__class__.__name__ + "("
        out += repr_module_like(
            self,
            module_keys=_LAYER_MODULE_LIKES,
            extra_keys=[v for v in self.__class__.__slots__ if v not in _LAYER_MODULE_LIKES],
        )
        out += ")"
        return out

    def __hash__(self) -> int:
        return hash((self.base, self.aggregate, self.transform, self.compilable))

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, Layer)
            and value.base == self.base
            and value.aggregate == self.aggregate
            and value.transform == self.transform
            and value.count == self.count
            and value.shape == self.shape
            and value.compilable == self.compilable
        )


class FactLayer:
    __slots__ = ("facts", "count", "shape")

    def __init__(self, facts: Sequence[Fact], count: int | None = None, shape: Shape | None = None) -> None:
        self.facts = facts
        self.count = count
        self.shape = shape if shape is not None else VariousShape()

    def __repr__(self) -> str:
        n = 3
        items_repr = ", ".join((fact_repr(f) for f in self.facts[:n]))
        if len(self.facts) > n:
            items_repr += f", ... (size: {len(self.facts)})"
        return f"{self.__class__.__name__}({items_repr}, count={self.count}, shape={my_repr(self.shape)})"

    def __hash__(self) -> int:
        return hash((tuple(self.facts), self.count, self.shape))

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, FactLayer)
            and len(self.facts) == len(value.facts)
            and all(((a == b) for a, b in zip(self.facts, value.facts)))
        )
