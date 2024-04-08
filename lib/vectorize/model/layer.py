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


class LinearLayerBase:
    __slots__ = ("input", "weight")
    __repr__ = repr_slots

    def __init__(self, input: Input, weight: Input) -> None:
        self.input = input
        self.weight = weight


class LinearGatherLayerBase:
    __slots__ = ("input", "weight", "gather")
    __repr__ = repr_slots

    def __init__(self, input: Input, weight: Input, gather: Gather) -> None:
        self.input = input
        self.weight = weight
        self.gather = gather


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
        out += repr_module_like(self, is_module=lambda k, _: k in _LAYER_MODULE_LIKES)
        out += ")"
        return out


class FactLayer:
    __slots__ = ("facts", "count", "shape")

    def __init__(self, facts: list[Fact], count: int | None = None, shape: Shape | None = None) -> None:
        self.facts = facts
        self.count = count
        self.shape = shape if shape is not None else VariousShape()

    def __repr__(self) -> str:
        n = 3
        items_repr = ", ".join((fact_repr(f) for f in self.facts[:n]))
        if len(self.facts) > n:
            items_repr += f", ... (size: {len(self.facts)})"
        return f"{self.__class__.__name__}({items_repr}, count={self.count}, shape={my_repr(self.shape)})"
