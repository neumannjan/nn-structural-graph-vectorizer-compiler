from typing import OrderedDict

from lib.nn.definitions.ops import ReductionDef
from lib.utils import addindent
from lib.vectorize.model.gather import (
    Gather,
    GatherPair,
    GenericGather,
    NoopGather,
    Repeat,
    SliceValues,
    TakeSingleValue,
)
from lib.vectorize.model.layer import FactLayer, GatheredLayers
from lib.vectorize.model.reduce import UnevenReduce
from lib.vectorize.model.repr import ModuleDictWrapper, repr_slots
from lib.vectorize.model.shape import ConcreteShape
from lib.vectorize.model.source import LayerRefs, RefPool
from lib.vectorize.model.transform import Transform
from lib.vectorize.model.weight import LearnableWeight


class Linear:
    __slots__ = ("weight_refs", "additional_ops")
    __repr__ = repr_slots

    def __init__(self, weight_refs: "LayerRefs", additional_ops: "OperationSeq") -> None:
        self.weight_refs = weight_refs
        self.additional_ops = additional_ops


class View:
    __slots__ = ("shape",)

    def __init__(self, shape: ConcreteShape) -> None:
        self.shape = shape

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.shape)})"


class DimReduce:
    __slots__ = ("dim", "reduce")
    __repr__ = repr_slots

    def __init__(self, dim: int, reduce: ReductionDef) -> None:
        self.dim = dim
        self.reduce: ReductionDef = reduce


Operation = Linear | Gather | Transform | DimReduce | UnevenReduce | LayerRefs | View


def _match_op(op: Operation):
    match op:
        case Linear(weight=weight):
            ...
        case GenericGather(ordinals=ordinals):
            ...
        case TakeSingleValue(ordinal=ordinal):
            ...
        case NoopGather():
            ...
        case SliceValues(start=start, end=end, step=step):
            ...
        case Repeat(times=_, total_length=total_length):
            ...
        case GatherPair(a, b):
            ...
        case Transform(transform=transform):
            ...
        case DimReduce(dim=dim, reduce=reduce):
            ...
        case UnevenReduce(counts=counts, reduce=reduce):
            ...
        case LayerRefs(facts=facts, weights=weights, layers=layers):
            ...
        case View(shape=shape):
            ...
        case _:
            assert False, f"{op}"


class OperationSeq:
    __slots__ = ("operations",)

    def __init__(self, operations: list[Operation]) -> None:
        self.operations = operations

    def __repr__(self) -> str:
        if len(self.operations) == 0:
            return self.__class__.__name__ + "()"

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
        batches: OrderedDict[int, OpSeqBatch],
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