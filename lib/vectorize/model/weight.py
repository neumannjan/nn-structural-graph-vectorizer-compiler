import numpy as np

from lib.vectorize.model.repr import repr_slots


class LearnableWeight:
    __slots__ = ("value",)
    __repr__ = repr_slots

    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, LearnableWeight)
            and self.value.shape == value.value.shape
            and bool(np.all(self.value == value.value))
        )
