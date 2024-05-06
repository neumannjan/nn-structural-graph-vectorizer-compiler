from typing import Literal

from lib.model.ops import ReductionDef
from lib.vectorize.model.noop import Noop
from lib.vectorize.model.repr import repr_module_like, repr_slots


class FixedCountReduce:
    __slots__ = ("period", "reduce", "dim")
    __repr__ = repr_slots

    def __init__(self, period: int, reduce: ReductionDef, dim: Literal[0, 1] = 1) -> None:
        self.period = period
        self.reduce: ReductionDef = reduce
        self.dim: Literal[0, 1] = dim

    def __hash__(self) -> int:
        return hash((self.period, self.reduce, self.dim))

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, FixedCountReduce) and self.period == value.period and self.reduce == value.reduce


class UnevenReduce:
    __slots__ = ("counts", "reduce")
    __repr__ = repr_slots

    def __init__(self, counts: list[int], reduce: ReductionDef) -> None:
        self.counts = counts
        self.reduce: ReductionDef = reduce

    def __hash__(self) -> int:
        return hash((tuple(self.counts), self.reduce))

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, UnevenReduce)
            and self.reduce == value.reduce
            and len(self.counts) == len(value.counts)
            and all((a == b for a, b in zip(self.counts, value.counts)))
        )

    @property
    def total(self) -> int:
        return sum(self.counts)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return name + "(" + repr_module_like(self, module_keys=[], extra_keys=["counts", "total", "reduce"]) + ")"


Reduce = FixedCountReduce | UnevenReduce | Noop


def _match_all(reduce: Reduce):
    match reduce:
        case FixedCountReduce(period=period, reduce=r, dim=dim):
            ...
        case UnevenReduce(counts=counts, reduce=r):
            ...
        case Noop():
            ...
        case _:
            assert False, f"{reduce}"
