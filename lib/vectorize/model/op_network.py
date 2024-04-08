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
from lib.vectorize.model.layer import FactLayer
from lib.vectorize.model.reduce import UnevenReduce
from lib.vectorize.model.refs import LayerRefs
from lib.vectorize.model.repr import ModuleDictWrapper, my_repr, repr_module_like, repr_slots
from lib.vectorize.model.shape import ConcreteShape
from lib.vectorize.model.transform import Transform
from lib.vectorize.model.weight import LearnableWeight


class Linear:
    __slots__ = ("weight_ops",)
    __repr__ = repr_slots

    def __init__(self, weight_ops: "OperationSeq") -> None:
        self.weight_ops = weight_ops


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


Operation = Linear | Gather | Transform | DimReduce | UnevenReduce | View


def _match_op(op: Operation):
    match op:
        case Linear(weight_ops=weight_ops):
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
        case View(shape=shape):
            ...
        case _:
            assert False, f"{op}"


class OperationSeq:
    __slots__ = ("layer_refs", "operations")

    def __init__(self, layer_refs: LayerRefs | None, operations: list[Operation]) -> None:
        self.layer_refs = layer_refs
        self.operations = operations

    def __repr__(self) -> str:
        operations = [self.layer_refs] + self.operations

        if len(operations) == 0:
            return self.__class__.__name__ + f"({self.layer_refs})"

        out_dict = {}
        out_dict["layer_refs"] = self.layer_refs
        for i in range(len(self.operations)):
            out_dict[i] = self.operations[i]

        return (
            self.__class__.__name__
            + "("
            + repr_module_like(out_dict, module_keys=["layer_refs"] + list(range(len(self.operations))), extra_keys=())
            + ")"
        )


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
    __slots__ = ("fact_layers", "weights", "batches")

    def __init__(
        self,
        fact_layers: dict[str, FactLayer],
        weights: dict[str, LearnableWeight],
        batches: OrderedDict[int, OpSeqBatch],
    ) -> None:
        self.fact_layers = fact_layers
        self.weights = weights
        self.batches = batches

    def __repr__(self) -> str:
        if not isinstance(self.fact_layers, ModuleDictWrapper):
            return repr_slots(
                self.__class__(
                    fact_layers=ModuleDictWrapper(self.fact_layers),  # pyright: ignore
                    weights=ModuleDictWrapper(self.weights),  # pyright: ignore
                    batches=ModuleDictWrapper(self.batches),  # pyright: ignore
                )
            )
        else:
            return repr_slots(self)
