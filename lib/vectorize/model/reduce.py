from lib.nn.definitions.ops import ReductionDef
from lib.vectorize.model.noop import Noop
from lib.vectorize.model.repr import repr_slots


class DimReduce:
    __slots__ = ("dim", "reduce")
    __repr__ = repr_slots

    def __init__(self, dim: int, reduce: ReductionDef) -> None:
        assert dim > 0, "DimReduce cannot have dim=0. Dim == 0 is reserved for count."
        self.dim = dim
        self.reduce: ReductionDef = reduce


class FixedCountReduce:
    __slots__ = ("period", "reduce")
    __repr__ = repr_slots

    def __init__(self, period: int, reduce: ReductionDef) -> None:
        self.period = period
        self.reduce: ReductionDef = reduce


class UnevenReduce:
    __slots__ = ("counts", "reduce")
    __repr__ = repr_slots

    def __init__(self, counts: list[int], reduce: ReductionDef) -> None:
        self.counts = counts
        self.reduce: ReductionDef = reduce


Reduce = DimReduce | FixedCountReduce | UnevenReduce | Noop


def _match_all(reduce: Reduce):
    match reduce:
        case FixedCountReduce(period=period, reduce=r):
            ...
        case DimReduce(dim=dim, reduce=r):
            ...
        case UnevenReduce(counts=counts, reduce=r):
            ...
        case Noop():
            ...
        case _:
            assert False, f"{reduce}"
