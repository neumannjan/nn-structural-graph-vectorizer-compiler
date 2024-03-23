from typing import OrderedDict

import numpy as np

from lib.vectorize.model.layer import Layer
from lib.vectorize.model.repr import ModuleDictWrapper, my_repr, repr_module_like, repr_slots
from lib.vectorize.model.shape import Shape, VariousShape


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
    __slots__ = ("facts", "shape")

    def __init__(self, facts: list[Fact], shape: Shape | None = None) -> None:
        self.facts = facts
        self.shape = shape if shape is not None else VariousShape()

    def __repr__(self) -> str:
        items_repr = ", ".join((_fact_repr(f) for f in self.facts[:3]))
        return f"{self.__class__.__name__}({items_repr}, ... (size: {len(self.facts)}))"


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
                    ModuleDictWrapper(self.layers),  # pyright: ignore
                )
            )
        else:
            return repr_slots(self)


class VectorizedNetwork:
    __slots__ = ("fact_layers", "weights", "batches")

    def __init__(
        self,
        fact_layers: dict[str, FactLayer],
        weights: dict[str, LearnableWeight],
        batches: dict[int, Batch],
    ) -> None:
        self.fact_layers = fact_layers
        self.weights = weights
        self.batches = batches

    def __repr__(self) -> str:
        if not isinstance(self.fact_layers, ModuleDictWrapper):
            return repr_slots(
                self.__class__(
                    ModuleDictWrapper(self.fact_layers),  # pyright: ignore
                    ModuleDictWrapper(self.weights),  # pyright: ignore
                    ModuleDictWrapper(self.batches),  # pyright: ignore
                )
            )
        else:
            return repr_slots(self)
