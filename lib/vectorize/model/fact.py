import numpy as np

from lib.vectorize.model.repr import my_repr, repr_slots


class ValueFact:
    __slots__ = ("value",)
    __repr__ = repr_slots

    def __init__(self, value: np.ndarray) -> None:
        self.value = value


class UnitFact:
    __slots__ = ()
    __repr__ = repr_slots


class EyeFact:
    __slots__ = ("dim",)
    __repr__ = repr_slots

    def __init__(self, dim: int) -> None:
        self.dim = dim


Fact = ValueFact | UnitFact | EyeFact


def _match_all_facts(fact: Fact):
    match fact:
        case UnitFact():
            ...
        case ValueFact(value=value):
            ...
        case EyeFact(dim=dim):
            ...
        case _:
            assert False, f"{fact}"


def fact_repr(fact: Fact) -> str:
    match fact:
        case ValueFact(value=value):
            return my_repr(list(value.shape))
        case UnitFact():
            return "unit"
        case EyeFact():
            return "eye"
        case _:
            assert False
