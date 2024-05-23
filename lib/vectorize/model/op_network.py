from typing import Iterator, OrderedDict, overload

from lib.model.ops import ReductionDef
from lib.vectorize.model.gather import (
    Gather,
    GatherPair,
    GenericGather,
    NoopGather,
    OneGather,
    Repeat,
    RepeatInterleave,
    SliceValues,
    TakeSingleValue,
)
from lib.vectorize.model.layer import FactLayer
from lib.vectorize.model.reduce import UnevenReduce
from lib.vectorize.model.refs import LayerRefs
from lib.vectorize.model.repr import ModuleDictWrapper, repr_module_like, repr_slots
from lib.vectorize.model.shape import ConcreteShape
from lib.vectorize.model.transform import Transform
from lib.vectorize.model.weight import LearnableWeight


class Linear:
    __slots__ = ("weight_ops",)
    __repr__ = repr_slots

    def __init__(self, weight_ops: "OperationSeq") -> None:
        self.weight_ops = weight_ops

    def __hash__(self) -> int:
        return hash(self.weight_ops)

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, Linear) and self.weight_ops == value.weight_ops


class View:
    __slots__ = ("shape",)

    def __init__(self, shape: ConcreteShape) -> None:
        self.shape = shape

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.shape)})"

    def __hash__(self) -> int:
        return hash(self.shape)

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, View) and self.shape == value.shape


class DimReduce:
    __slots__ = ("dim", "reduce")
    __repr__ = repr_slots

    def __init__(self, dim: int, reduce: ReductionDef) -> None:
        self.dim = dim
        self.reduce: ReductionDef = reduce

    def __hash__(self) -> int:
        return hash((self.dim, self.reduce))

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, DimReduce) and self.dim == value.dim and self.reduce == value.reduce


SimpleOperation = OneGather | Transform | DimReduce | UnevenReduce | View
Operation = Linear | Gather | Transform | DimReduce | UnevenReduce | View


class OperationSeq:
    __slots__ = ("layer_refs", "operations", "expected_count")

    def __init__(self, layer_refs: LayerRefs, operations: list[Operation], count: int | None) -> None:
        self.layer_refs = layer_refs
        self.operations = operations
        self.expected_count = count

    @overload
    def __getitem__(self, key: int) -> "Operation": ...

    @overload
    def __getitem__(self, key: slice) -> "list[Operation]": ...

    def __getitem__(self, key: int | slice) -> "Operation | list[Operation]":
        return self.operations[key]

    def __repr__(self) -> str:
        operations = [self.layer_refs] + self.operations

        if len(operations) == 0:
            return self.__class__.__name__ + f"({self.layer_refs})"

        return (
            self.__class__.__name__
            + "("
            + repr_module_like(
                self, module_keys=["layer_refs"] + list(range(len(self.operations))), extra_keys=("expected_count",)
            )
            + ")"
        )

    def __iter__(self) -> Iterator[Operation]:
        return iter(self.operations)

    def __hash__(self) -> int:
        return hash((self.layer_refs, tuple(self.operations)))

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, OperationSeq)
            and self.layer_refs == value.layer_refs
            and all((o1 == o2) for o1, o2 in zip(self.operations, value.operations))
            and (
                self.expected_count == value.expected_count
                or self.expected_count is None
                or value.expected_count is None
            )
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

    def __hash__(self) -> int:
        return hash(tuple(((k, v) for k, v in self.layers.items())))

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, OpSeqBatch) and self.layers == value.layers


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

    def __hash__(self) -> int:
        fl_keys = sorted(self.fact_layers)
        w_keys = sorted(self.weights)

        return hash(
            (
                tuple(((k, self.fact_layers[k]) for k in fl_keys)),
                tuple(((k, self.weights[k]) for k in w_keys)),
                tuple(((k, v) for k, v in self.batches.items())),
            )
        )

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, VectorizedOpSeqNetwork)
            and self.fact_layers == value.fact_layers
            and self.weights == value.weights
            and self.batches == value.batches
        )
