from typing import OrderedDict

from lib.nn.definitions.ops import ReductionDef
from lib.utils import addindent
from lib.vectorize.model.gather import Gather
from lib.vectorize.model.layer import FactLayer, GatheredLayers
from lib.vectorize.model.reduce import Reduce, UnevenReduce
from lib.vectorize.model.repr import ModuleDictWrapper, repr_slots
from lib.vectorize.model.shape import ConcreteShape
from lib.vectorize.model.source import LayerRefs, RefPool
from lib.vectorize.model.transform import Transform
from lib.vectorize.model.weight import LearnableWeight


class Linear:
    __slots__ = ("weight",)
    __repr__ = repr_slots

    def __init__(self, weight: GatheredLayers) -> None:
        self.weight = weight


class View:
    __slots__ = ("shape",)
    __repr__ = repr_slots

    def __init__(self, shape: ConcreteShape) -> None:
        self.shape = shape


class DimReduce:
    __slots__ = ("dim", "reduce")
    __repr__ = repr_slots

    def __init__(self, dim: int, reduce: ReductionDef) -> None:
        self.dim = dim
        self.reduce = reduce


Operation = Linear | Gather | Transform | DimReduce | UnevenReduce | LayerRefs | View


class OperationSeq:
    __slots__ = ("operations",)

    def __init__(self, operations: list[Operation]) -> None:
        self.operations = operations

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(\n  " + addindent(",\n".join((repr(o) for o in self.operations)), 2) + "\n)"


class OpSeqBatch:
    __slots__ = ("layers",)
    __repr__ = repr_slots

    def __init__(self, layers: OrderedDict[str, OperationSeq]) -> None:
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


class VectorizedOpSeqNetwork:
    __slots__ = ("fact_layers", "weights", "batches", "ref_pool")

    def __init__(
        self,
        fact_layers: dict[str, FactLayer],
        weights: dict[str, LearnableWeight],
        batches: dict[int, OpSeqBatch],
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
