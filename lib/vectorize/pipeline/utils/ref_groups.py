from typing import Hashable, Iterable, Sequence, TypeVar

from lib.vectorize.model import *

_TRef = TypeVar("_TRef", bound=Hashable)


def _grouper(iterable: Iterable[_TRef], n: int) -> Iterable[tuple[_TRef, ...]]:
    it = iter(iterable)
    while True:
        try:
            yield tuple([next(it) for _ in range(n)])
        except StopIteration:
            break


def _uneven_grouper(iterable: Iterable[_TRef], counts: list[int]) -> Iterable[tuple[_TRef, ...]]:
    it = iter(iterable)

    for c in counts:
        yield tuple((next(it) for _ in range(c)))


def get_refs(source: Refs | GenericGather):
    match source:
        case Refs():
            return source
        case GenericGather():
            return source.ordinals
        case _:
            assert False, f"{source}"


def get_ref_groups(aggregate: Reduce, refs: Sequence[_TRef]) -> Sequence[tuple[_TRef, ...]]:
    match aggregate:
        case Noop():
            return [(r,) for r in refs]
        case FixedCountReduce(period=period):
            return list(_grouper(refs, n=period))
        case UnevenReduce(counts=counts):
            return list(_uneven_grouper(refs, counts=counts))
        case _:
            assert False, f"{aggregate}"
