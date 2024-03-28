import numpy as np

from lib.vectorize.model.repr import repr_slots


class LearnableWeight:
    __slots__ = ("value",)
    __repr__ = repr_slots

    def __init__(self, value: np.ndarray) -> None:
        self.value = value
