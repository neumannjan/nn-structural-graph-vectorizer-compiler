import hashlib

import numpy as np


class HashableArray:
    def __init__(self, arr: np.ndarray) -> None:
        self.arr = arr
        self._hash: int | None = None

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = out = hash(hashlib.sha256(self.arr.data, usedforsecurity=False).digest())
            return out

        return self._hash

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, HashableArray)
            and self.arr.shape == value.arr.shape
            and bool(np.all(self.arr == value.arr))
        )
