from collections.abc import Sequence

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

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, GatheredLayers) and self.refs == value.refs and self.gather == value.gather


Input = GatheredLayers | Refs


def _match_input(input: Input):
    match input:
        case Refs():
            ...
        case GatheredLayers(refs=refs, gather=gather):
            ...
        case _:
            assert False, f"{input}"


class InputLayerBase:
    __slots__ = ("input",)
    __repr__ = repr_slots

    def __init__(self, input: Input) -> None:
        self.input = input

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, InputLayerBase) and self.input == value.input


class LinearLayerBase:
    __slots__ = ("input", "weight")
    __repr__ = repr_slots

    def __init__(self, input: Input, weight: Input) -> None:
        self.input = input
        self.weight = weight

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, LinearLayerBase) and self.input == value.input and self.weight == value.weight


class LinearGatherLayerBase:
    __slots__ = ("input", "weight", "gather")
    __repr__ = repr_slots

    def __init__(self, input: Input, weight: Input, gather: Gather) -> None:
        self.input = input
        self.weight = weight
        self.gather = gather

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, LinearGatherLayerBase)
            and self.input == value.input
            and self.weight == value.weight
            and self.gather == value.gather
        )


LayerBase = InputLayerBase | LinearLayerBase | LinearGatherLayerBase


def _match_layer_base(base: LayerBase):
    match base:
        case InputLayerBase(input=input):
            ...
        case LinearLayerBase(input=input, weight=weight):
            ...
        case LinearGatherLayerBase(input=input, weight=weight, gather=gather):
            ...
        case _:
            assert False


_LAYER_MODULE_LIKES = ("base", "aggregate", "transform")


class Layer:
    __slots__ = ("base", "aggregate", "transform", "count", "shape", "ord_map")
    __repr__ = repr_slots

    def __init__(
        self,
        base: LayerBase,
        aggregate: Reduce,
        transform: Transform,
        count: int | None = None,
        shape: Shape | None = None,
    ) -> None:
        self.base = base
        self.aggregate = aggregate
        self.transform = transform
        self.count = count
        self.shape = shape if shape is not None else VariousShape()
        self.ord_map: dict[int, int] = {}

    def __repr__(self) -> str:
        out = self.__class__.__name__ + "("
        out += repr_module_like(
            self,
            module_keys=_LAYER_MODULE_LIKES,
            extra_keys=[v for v in self.__class__.__slots__ if v not in _LAYER_MODULE_LIKES],
        )
        out += ")"
        return out

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, Layer)
            and value.base == self.base
            and value.aggregate == self.aggregate
            and value.transform == self.transform
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

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, FactLayer)
            and len(self.facts) == len(value.facts)
            and all(((a == b) for a, b in zip(self.facts, value.facts)))
        )
