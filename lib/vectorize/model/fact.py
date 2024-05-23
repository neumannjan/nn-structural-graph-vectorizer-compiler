import numpy as np

from lib.vectorize.model.repr import my_repr, repr_slots


class ValueFact:
    __slots__ = ("value",)
    __repr__ = repr_slots

    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def __hash__(self) -> int:
        return hash(tuple(self.value.flatten().tolist()))

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, ValueFact) and np.array_equal(self.value, value.value, equal_nan=True)


class UnitFact:
    __slots__ = ()
    __repr__ = repr_slots

    def __hash__(self) -> int:
        return hash(())

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, UnitFact)


class EyeFact:
    __slots__ = ("dim",)
    __repr__ = repr_slots

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def __hash__(self) -> int:
        return hash(self.dim)

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, EyeFact) and self.dim == value.dim


Fact = ValueFact | UnitFact | EyeFact


def fact_repr(fact: Fact) -> str:
    if isinstance(fact, ValueFact):
        return my_repr(list(fact.value.shape))
    elif isinstance(fact, UnitFact):
        return "unit"
    elif isinstance(fact, EyeFact):
        return "eye"
    else:
        assert False
