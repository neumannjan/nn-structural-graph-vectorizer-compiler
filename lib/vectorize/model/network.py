from typing import OrderedDict

import numpy as np

from lib.vectorize.model.layer import Layer
from lib.vectorize.model.repr import ModuleDictWrapper, my_repr, repr_slots
from lib.vectorize.model.shape import Shape, VariousShape
from lib.vectorize.model.source import RefPool


class ValueFact:
    __slots__ = ("value",)
    __repr__ = repr_slots

    def __init__(self, value: np.ndarray) -> None:
        self.value = value


class UnitFact:
    __slots__ = ()
    __repr__ = repr_slots


Fact = ValueFact | UnitFact


def _match_all_facts(fact: Fact):
    match fact:
        case UnitFact():
            ...
        case ValueFact(value=value):
            ...
        case _:
            assert False, f"{fact}"


def _fact_repr(fact: Fact) -> str:
    match fact:
        case ValueFact(value=value):
            return my_repr(list(value.shape))
        case UnitFact():
            return "unit"
        case _:
            assert False


class FactLayer:
    __slots__ = ("facts", "count", "shape")

    def __init__(self, facts: list[Fact], count: int | None = None, shape: Shape | None = None) -> None:
        self.facts = facts
        self.count = count
        self.shape = shape if shape is not None else VariousShape()

    def __repr__(self) -> str:
        items_repr = ", ".join((_fact_repr(f) for f in self.facts[:3]))
        return f"{self.__class__.__name__}({items_repr}, ... (size: {len(self.facts)}), count={self.count}, shape={my_repr(self.shape)})"


class LearnableWeight:
    __slots__ = ("value",)
    __repr__ = repr_slots

    def __init__(self, value: np.ndarray) -> None:
        self.value = value


class Batch:
    __slots__ = ("layers",)
    __repr__ = repr_slots

    def __init__(self, layers: OrderedDict[str, Layer]) -> None:
        self.layers = layers

    def __repr__(self) -> str:
        if not isinstance(self.layers, ModuleDictWrapper):
            return repr_slots(
                self.__class__(
                    layers=ModuleDictWrapper(self.layers),  # pyright: ignore
                )
            )
        else:
            return repr_slots(self)


class VectorizedNetwork:
    __slots__ = ("fact_layers", "weights", "batches", "ref_pool")

    def __init__(
        self,
        fact_layers: dict[str, FactLayer],
        weights: dict[str, LearnableWeight],
        batches: dict[int, Batch],
        ref_pool: RefPool,
    ) -> None:
        self.fact_layers = fact_layers
        self.weights = weights
        self.batches = batches
        self.ref_pool = ref_pool

    def __repr__(self) -> str:
        if not isinstance(self.fact_layers, ModuleDictWrapper):
            return repr_slots(
                self.__class__(
                    fact_layers=ModuleDictWrapper(self.fact_layers),  # pyright: ignore
                    weights=ModuleDictWrapper(self.weights),  # pyright: ignore
                    batches=ModuleDictWrapper(self.batches),  # pyright: ignore
                    ref_pool=self.ref_pool,
                )
            )
        else:
            return repr_slots(self)
