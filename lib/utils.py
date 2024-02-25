import functools
import heapq
from collections.abc import Collection
from typing import Callable, Generic, Iterable, Iterator, Sequence, TypeVar

import numpy as np
import torch

DTYPE_TORCH_TO_NUMPY = {
    torch.float32: np.float32,
    torch.float64: np.float64,
}


def value_to_numpy(java_value, dtype: torch.dtype | None = None) -> np.ndarray:
    if dtype is None:
        dtype = torch.get_default_dtype()

    if dtype not in DTYPE_TORCH_TO_NUMPY:
        raise NotImplementedError(f"Conversion from {dtype} to numpy equivalent not yet implemented.")

    np_dtype = DTYPE_TORCH_TO_NUMPY[dtype]

    arr = np.asarray(java_value.getAsArray(), dtype=np_dtype)
    arr = arr.reshape(java_value.size())
    return arr


def value_to_tensor(java_value, dtype: torch.dtype | None = None) -> torch.Tensor:
    return torch.tensor(value_to_numpy(java_value, dtype))


def atleast_3d_rev(tensor: torch.Tensor) -> torch.Tensor:
    dim = tensor.dim()

    if dim == 0:
        return tensor.view([1, 1, 1])
    elif dim == 1:
        return tensor.view([-1, 1, 1])
    elif dim == 2:
        return tensor.unsqueeze(-1)
    else:
        return tensor


def atleast_2d_rev(tensor: torch.Tensor) -> torch.Tensor:
    dim = tensor.dim()

    if dim == 0:
        return tensor.view([1, 1])
    elif dim == 1:
        return tensor.view([-1, 1])
    else:
        return tensor


def _argmax_last(a: np.ndarray):
    a = a[::-1]
    return len(a) - np.argmax(a) - 1


def _detect_possible_last_incomplete(inp: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    # find last occurrence of the initial string
    last_idx = _argmax_last(inp == inp[0])

    if last_idx <= len(inp) // 2:
        return inp, None

    return inp[:last_idx], inp[last_idx:]


def cache(func):
    @functools.wraps(func)
    @functools.lru_cache(maxsize=None)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


_T = TypeVar("_T")
_S = TypeVar("_S")


class KeySortable(Generic[_T, _S]):
    __slots__ = "key", "value"

    def __init__(self, key: _T, value: _S) -> None:
        self.key = key
        self.value = value

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, KeySortable) and self.key == other.key

    def __lt__(self, other: "KeySortable") -> bool:
        return self.key < other.key

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({self.key}, {self.value})"


def detect_repeating_sequence_in_list(
    inp: Sequence[int] | np.ndarray, allow_last_incomplete=False
) -> np.ndarray | None:
    """
    Find the smallest repeating subsequence `s` that the input sequence `t` is made from (such that `t = ss...s`).

    Returns None if the only such subsequence is the original sequence `t` itself (i.e. when the sequence should not be
    categorized as a repeated smaller sequence).
    If `allow_last_incomplete` is `True`, allows the final repetition to be incomplete.

    >>> detect_repeating_sequence_in_list([0], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([0, 1], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([0, 1, 2], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1, 2], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2], allow_last_incomplete=False)
    array([0, 1, 2, 2])
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2], allow_last_incomplete=False)
    array([0, 1, 2, 2])
    >>> detect_repeating_sequence_in_list([0], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([0, 1], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([0, 1, 2], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0], allow_last_incomplete=True)
    array([0, 1, 2, 2])
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1], allow_last_incomplete=True)
    array([0, 1, 2, 2])
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1, 2], allow_last_incomplete=True)
    array([0, 1, 2, 2])
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2], allow_last_incomplete=True)
    array([0, 1, 2, 2])
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0], allow_last_incomplete=True)
    array([0, 1, 2, 2])
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1], allow_last_incomplete=True)
    array([0, 1, 2, 2])
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2], allow_last_incomplete=True)
    array([0, 1, 2, 2])
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2], allow_last_incomplete=True)
    array([0, 1, 2, 2])
    >>> detect_repeating_sequence_in_list([0], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([0, 0], allow_last_incomplete=False)
    array([0])
    >>> detect_repeating_sequence_in_list([0, 0, 0], allow_last_incomplete=False)
    array([0])
    >>> detect_repeating_sequence_in_list([0], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([0, 0], allow_last_incomplete=True)
    array([0])
    >>> detect_repeating_sequence_in_list([0, 0, 0], allow_last_incomplete=True)
    array([0])
    >>> detect_repeating_sequence_in_list([2, 0], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([2, 0, 1], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2], allow_last_incomplete=False)
    array([2, 0, 1, 2])
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2], allow_last_incomplete=False)
    array([2, 0, 1, 2])
    >>> detect_repeating_sequence_in_list([2, 0], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([2, 0, 1], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2], allow_last_incomplete=True)
    array([2, 0, 1])
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2], allow_last_incomplete=True)
    array([2, 0, 1, 2])
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0], allow_last_incomplete=True)
    array([2, 0, 1, 2])
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1], allow_last_incomplete=True)
    array([2, 0, 1, 2])
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2], allow_last_incomplete=True)
    array([2, 0, 1, 2])
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2], allow_last_incomplete=True)
    array([2, 0, 1, 2])
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0], allow_last_incomplete=True)
    array([2, 0, 1, 2])
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1], allow_last_incomplete=True)
    array([2, 0, 1, 2])
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2], allow_last_incomplete=True)
    array([2, 0, 1, 2])
    >>> detect_repeating_sequence_in_list([2, 0, 1, 3], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 3], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 3], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 3], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 3], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 3], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 3], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 3], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 3], allow_last_incomplete=True)
    """
    inp = np.array(inp)

    # we need to find possible period lengths (K):

    # find only Ks up to the middle of the sequence (because the string is repeated at least twice), which are also
    # equal to the initial value
    (candidate_Ks,) = np.where(inp[: len(inp) // 2 + 1] == inp[0])

    # skip the index 0
    candidate_Ks = candidate_Ks[1:]

    # select only Ks divisible by the full length
    candidate_Ks = candidate_Ks[len(inp) % candidate_Ks == 0]

    candidate_Ks = candidate_Ks.tolist()

    candidate_pairs: Iterable[KeySortable[int, tuple[np.ndarray, np.ndarray | None]]] = [
        KeySortable(v, (inp, None)) for v in candidate_Ks
    ]

    if allow_last_incomplete:
        inp_first, inp_last = _detect_possible_last_incomplete(inp)
        if inp_last is not None:
            # find only Ks equal to the initial value
            (candidate_Ks,) = np.where(inp_first == inp_first[0])

            # skip the index 0
            candidate_Ks = candidate_Ks[1:]

            # select only Ks divisible by the full length
            candidate_Ks = candidate_Ks[len(inp_first) % candidate_Ks == 0]

            candidate_Ks = candidate_Ks.tolist()

            # add full length option
            candidate_Ks.append(len(inp_first))

            candidate_pairs2: list[KeySortable[int, tuple[np.ndarray, np.ndarray | None]]] = [
                KeySortable(v, (inp_first, inp_last)) for v in candidate_Ks
            ]

            # both lists are already sorted, so we can use an O(n) algorithm
            candidate_pairs = heapq.merge(candidate_pairs, candidate_pairs2)
    else:
        inp_last = None

    # check all remaining Ks for periodicity, starting with the shortest
    for candidate in candidate_pairs:
        k = candidate.key
        inp_first, inp_last = candidate.value

        if np.all(inp_first.reshape([-1, k]) == inp_first[:k]) and (
            inp_last is None or np.all(inp_last == inp_first[: len(inp_last)])
        ):
            return inp_first[:k]

    return None


def head_and_rest(it: Iterator[_T] | Iterable[_T]) -> tuple[_T, Iterator[_T]]:
    if not isinstance(it, Iterator):
        it = iter(it)

    head = next(it)
    return head, it


class MapCollection(Collection[_T]):
    def __init__(self, mapping: Callable[[_S], _T], orig: Collection[_S]) -> None:
        super().__init__()
        self._orig = orig
        self._mapping = mapping

    def __len__(self) -> int:
        return len(self._orig)

    def __iter__(self) -> Iterator[_T]:
        return map(self._mapping, self._orig)

    def __contains__(self, x: object) -> bool:
        for y in self:
            if y == x:
                return True

        return False


class LambdaIterable(Iterable[_T]):
    def __init__(self, func: Callable[[], Iterator[_T]]) -> None:
        self._func = func

    def __iter__(self) -> Iterator[_T]:
        return self._func()


def print_with_ellipsis(it: Iterator[str], after=5) -> str:
    vals: list[str] = []

    for i in range(after):
        try:
            val = next(it)
            vals.append(str(val))
        except StopIteration:
            return ", ".join(vals)

    try:
        val = next(it)
    except StopIteration:
        return ", ".join(vals)

    vals.append("...")
    return ", ".join(vals)
