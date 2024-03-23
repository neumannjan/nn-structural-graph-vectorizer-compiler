from typing import Sequence

from lib.utils import cache
from lib.vectorize.model.repr import my_repr, repr_slots


class FactRef:
    __slots__ = ("id", "ordinal")
    __repr__ = repr_slots

    def __init__(self, id: str, ordinal: int) -> None:
        self.id = id
        self.ordinal = ordinal

    def __hash__(self) -> int:
        return hash((self.id, self.ordinal))

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, FactRef) and self.id == value.id and self.ordinal == value.ordinal

    def __lt__(self, ref: "Ref"):
        if isinstance(ref, FactRef):
            return self.id < ref.id or (self.id == ref.id and self.ordinal < ref.ordinal)
        return True


class WeightRef:
    __slots__ = ("id",)
    __repr__ = repr_slots

    def __init__(self, id: str) -> None:
        self.id = id

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, WeightRef) and self.id == value.id

    def __lt__(self, ref: "Ref | LayerRef"):
        if isinstance(ref, (FactRef, FactLayerRef)):
            return False

        if isinstance(ref, WeightRef):
            return self.id < ref.id

        return True


class NeuronRef:
    __slots__ = ("id", "ordinal")
    __repr__ = repr_slots

    def __init__(self, id: str, ordinal: int) -> None:
        self.id = id
        self.ordinal = ordinal

    def __hash__(self) -> int:
        return hash((self.id, self.ordinal))

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, NeuronRef) and self.id == value.id and self.ordinal == value.ordinal

    def __lt__(self, ref: "Ref"):
        if isinstance(ref, (FactRef, WeightRef)):
            return False

        if isinstance(ref, NeuronRef):
            return self.id < ref.id or (self.id == ref.id and self.ordinal < ref.ordinal)

        assert False


Ref = FactRef | NeuronRef | WeightRef


def _match_all_ref(ref: Ref):
    match ref:
        case FactRef(id=id, ordinal=ordinal):
            ...
        case NeuronRef(id=id, ordinal=ordinal):
            ...
        case WeightRef(id=id):
            ...
        case _:
            assert False, f"{ref}"


class Refs:
    __slots__ = ("refs",)
    __match_args__ = ("refs",)

    def __init__(self, refs: list["Ref"]) -> None:
        self.refs = refs

    def __hash__(self) -> int:
        return hash(iter(self.refs))

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, Refs)
            and len(self.refs) == len(value.refs)
            and all((a == b for a, b in zip(self.refs, value.refs)))
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({my_repr(self.refs)})"


class FactLayerRef:
    __slots__ = ("id",)
    __repr__ = repr_slots

    def __init__(self, id: str) -> None:
        self.id = id

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, FactLayerRef) and self.id == value.id

    def __lt__(self, ref: "LayerRef"):
        if isinstance(ref, FactLayerRef):
            return self.id < ref.id

        return True


class NeuronLayerRef:
    __slots__ = ("id",)
    __repr__ = repr_slots

    def __init__(self, id: str) -> None:
        self.id = id

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, FactLayerRef) and self.id == value.id

    def __lt__(self, ref: "LayerRef"):
        if isinstance(ref, FactLayerRef):
            return False

        if isinstance(ref, NeuronLayerRef):
            return self.id < ref.id

        return True


LayerRef = FactLayerRef | NeuronLayerRef | WeightRef


def _match_all_layer_ref(ref: LayerRef):
    match ref:
        case FactLayerRef(id=id):
            ...
        case NeuronLayerRef(id=id):
            ...
        case WeightRef(id=id):
            ...
        case _:
            assert False, f"{ref}"


class LayerRefs:
    __slots__ = ("refs",)
    __match_args__ = ("refs",)

    def __init__(self, refs: Sequence["LayerRef"]) -> None:
        self.refs = refs

    def __hash__(self) -> int:
        return hash(iter(self.refs))

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, Refs)
            and len(self.refs) == len(value.refs)
            and all((a == b for a, b in zip(self.refs, value.refs)))
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({my_repr(self.refs)})"


class RefPool:
    @cache
    def fact(self, id: str, ordinal: int):
        return FactRef(id, ordinal)

    @cache
    def neuron(self, id: str, ordinal: int):
        return NeuronRef(id, ordinal)

    @cache
    def weight(self, id: str):
        return WeightRef(id)


class LayerRefPool:
    @cache
    def fact(self, id: str):
        return FactLayerRef(id)

    @cache
    def neuron(self, id: str):
        return NeuronLayerRef(id)
