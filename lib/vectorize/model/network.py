from typing import OrderedDict

from lib.vectorize.model.layer import FactLayer, Layer
from lib.vectorize.model.repr import ModuleDictWrapper, repr_slots
from lib.vectorize.model.source import RefPool
from lib.vectorize.model.weight import LearnableWeight


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
