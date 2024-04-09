import functools
import heapq
import inspect
from collections.abc import Collection
from types import MethodType
from typing import (
    Callable,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    MutableSet,
    OrderedDict,
    Sequence,
    Type,
    TypeVar,
    overload,
)

import numpy as np
import torch


def atleast_2d_shape(shape: Sequence[int]) -> Sequence[int]:
    dim = len(shape)

    if dim == 0:
        return [1, 1]
    elif dim == 1:
        return [shape[0], 1]
    else:
        return shape


def atleast_3d_shape(shape: Sequence[int]) -> Sequence[int]:
    dim = len(shape)

    if dim == 0:
        return [1, 1, 1]
    elif dim == 1:
        return [shape[0], 1, 1]
    elif dim == 2:
        return [*shape, 1]
    else:
        return shape


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
_R = TypeVar("_R")


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


def detect_repeating_K_sequence_in_list(
    inp: Sequence | np.ndarray, period: int, allow_last_incomplete=False
) -> np.ndarray | None:
    """
    Find the smallest repeating subsequence `s` of length `period` that the input sequence `t` is made from.

    Find the smallest repeating subsequence `s` of length `period` that the input sequence `t` is made from
    (such that `t = ss...s`).

    Returns None if no such subsequence exists or if the only such subsequence
    is the original sequence `t` itself.
    If `allow_last_incomplete` is `True`, allows the final repetition to be incomplete.

    >>> detect_repeating_K_sequence_in_list([0], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 1], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2], 4, allow_last_incomplete=False)
    array([0, 1, 2, 2])
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2], 4, allow_last_incomplete=False)
    array([0, 1, 2, 2])
    >>> detect_repeating_K_sequence_in_list([0], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 1], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0], 4, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([0, 1], 4, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2], 4, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2], 4, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0], 4, allow_last_incomplete=True)
    array([0, 1, 2, 2])
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1], 4, allow_last_incomplete=True)
    array([0, 1, 2, 2])
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2], 4, allow_last_incomplete=True)
    array([0, 1, 2, 2])
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2], 4, allow_last_incomplete=True)
    array([0, 1, 2, 2])
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0], 4, allow_last_incomplete=True)
    array([0, 1, 2, 2])
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1], 4, allow_last_incomplete=True)
    array([0, 1, 2, 2])
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2], 4, allow_last_incomplete=True)
    array([0, 1, 2, 2])
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2], 4, allow_last_incomplete=True)
    array([0, 1, 2, 2])
    >>> detect_repeating_K_sequence_in_list([0], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([0, 1], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([0], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 0], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 0, 0], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 0, 0, 0], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 0, 0, 0, 0], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 0, 0, 0, 0, 0], 3, allow_last_incomplete=False)
    array([0, 0, 0])
    >>> detect_repeating_K_sequence_in_list([0, 0, 0, 0, 0, 0, 0], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([0, 0], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([0, 0, 0], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([0, 0, 0, 0], 3, allow_last_incomplete=True)
    array([0, 0, 0])
    >>> detect_repeating_K_sequence_in_list([0, 0, 0, 0, 0], 3, allow_last_incomplete=True)
    array([0, 0, 0])
    >>> detect_repeating_K_sequence_in_list([0, 0, 0, 0, 0, 0], 3, allow_last_incomplete=True)
    array([0, 0, 0])
    >>> detect_repeating_K_sequence_in_list([0, 0, 0, 0, 0, 0, 0], 3, allow_last_incomplete=True)
    array([0, 0, 0])
    >>> detect_repeating_K_sequence_in_list([2, 0], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2], 4, allow_last_incomplete=False)
    array([2, 0, 1, 2])
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2], 4, allow_last_incomplete=False)
    array([2, 0, 1, 2])
    >>> detect_repeating_K_sequence_in_list([2, 0], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0], 4, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1], 4, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2], 4, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2], 4, allow_last_incomplete=True)
    array([2, 0, 1, 2])
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0], 4, allow_last_incomplete=True)
    array([2, 0, 1, 2])
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1], 4, allow_last_incomplete=True)
    array([2, 0, 1, 2])
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2], 4, allow_last_incomplete=True)
    array([2, 0, 1, 2])
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2], 4, allow_last_incomplete=True)
    array([2, 0, 1, 2])
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0], 4, allow_last_incomplete=True)
    array([2, 0, 1, 2])
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1], 4, allow_last_incomplete=True)
    array([2, 0, 1, 2])
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2], 4, allow_last_incomplete=True)
    array([2, 0, 1, 2])
    >>> detect_repeating_K_sequence_in_list([2, 0], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2], 3, allow_last_incomplete=True)
    array([2, 0, 1])
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 3], 4, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 3], 4, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 3], 4, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 3], 4, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 3], 4, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 3], 4, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 3], 4, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 3], 4, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 3], 4, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 3], 4, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([['1', '14', 0], ['1', '15', 0], ['0', 'unit', '0'],
    ... ['1', '14', 0], ['1', '15', 0], ['0', 'unit', '0']], 3)
    array([['1', '14', '0'],
           ['1', '15', '0'],
           ['0', 'unit', '0']], dtype='<U21')
    >>> detect_repeating_K_sequence_in_list([['1', '14', 0], ['1', '15', 0], ['0', 'unit', '0'],
    ... ['1', '14', 0], ['1', '15', 0]], 3, allow_last_incomplete=True)
    array([['1', '14', '0'],
           ['1', '15', '0'],
           ['0', 'unit', '0']], dtype='<U21')
    """
    k = period

    if len(inp) <= k:
        return None

    if not isinstance(inp, np.ndarray):
        inp = np.array(inp)
    max_len = (inp.shape[0] // k) * k

    if not allow_last_incomplete and max_len < inp.shape[0]:
        return None

    inp_first, inp_last = inp[:max_len], inp[max_len:]

    if np.all(inp_first.reshape([-1, k, *inp_first.shape[1:]]) == inp_first[:k]) and (
        len(inp_last) == 0 or np.all(inp_last == inp_first[: len(inp_last)])
    ):
        return inp_first[:k]

    return None


def head_and_rest(it: Iterator[_T] | Iterable[_T]) -> tuple[_T, Iterator[_T]]:
    if not isinstance(it, Iterator):
        it = iter(it)

    head = next(it)
    return head, it


class MapCollection(Collection[_T]):
    __slots__ = ("_mapping", "_orig")

    def __init__(self, mapping: Callable[[_S], _T], orig: Collection[_S]) -> None:
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


class MapSequence(Sequence[_T]):
    __slots__ = ("_mapping", "_orig")

    def __init__(self, mapping: Callable[[_S], _T], orig: Sequence[_S]) -> None:
        self._orig = orig
        self._mapping = mapping

    @overload
    def __getitem__(self, key: int) -> _T: ...

    @overload
    def __getitem__(self, key: slice) -> Sequence[_T]: ...

    def __getitem__(self, key: int | slice) -> _T | Sequence[_T]:
        if isinstance(key, slice):
            return MapSequence(self._mapping, self._orig[key])

        return self._mapping(self._orig[key])

    def __iter__(self) -> Iterator[_T]:
        return map(self._mapping, self._orig)

    def __len__(self) -> int:
        return len(self._orig)


class MapMapping(Mapping[_T, _S]):
    __slots__ = ("_mapping", "_orig")

    def __init__(self, mapping: Callable[[_R], _S], orig: Mapping[_T, _R]) -> None:
        self._orig = orig
        self._mapping = mapping

    def __getitem__(self, key: _T) -> _S:
        return self._mapping(self._orig[key])

    def __iter__(self) -> Iterator[_T]:
        return iter(self._orig)

    def __len__(self) -> int:
        return len(self._orig)


class LambdaIterable(Iterable[_T]):
    __slots__ = ("_func",)

    def __init__(self, func: Callable[[], Iterator[_T]]) -> None:
        self._func = func

    def __iter__(self) -> Iterator[_T]:
        return self._func()


def print_with_ellipsis(it: Iterator[str], after=5) -> str:
    vals: list[str] = []

    for _ in range(after):
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


class ExtendsDynamicError(Exception):
    pass


class InheritDynamic:
    """
    Mark class property as a dynamically inherited method.

    Should be used with `@extends_dynamic` decorator. Please see its documentation.
    """

    def __get__(self, obj, objtype=None):
        cls_name = obj.__class__.__name__
        this_name = self.__class__.__name__
        decorator_name = extends_dynamic.__name__
        raise ExtendsDynamicError(
            f"Class {cls_name}: Cannot use {this_name} without @{decorator_name} decorator on the class!"
        )


def extends_dynamic(origin_property: str):
    """
    Decorate class to modify it to act as if it extends the instance passed in constructor.

    Each InheritDynamic method is replaced with the exact implementation from the origin.

    Example:
    ```
    class Container:
        def get_a(self):
            return "a"

        def get_b(self):
            return "b"

        def get_ab(self):
            return self.get_a() + self.get_b()

    @extends_dynamic("origin")
    class ContainerWrapper(Container):
        def __init__(self, origin: Container):
            self.origin = origin

        get_b = InheritDynamic()
        get_ab = InheritDynamic()

        def get_a(self):
            return "A"
    ```

    Calling `get_ab()` on `ContainerWrapper` returns `"Ab"`. It calls the original `self.origin.get_ab()` method,
    but with a modified `self`, such that `self.get_a()` resolves to the modified `get_a()` in the `ContainerWrapper`.

    Calling `origin.get_ab()` on `ContainerWrapper` returns `"ab"` still (the origin property itself isn't modified).

    WARNING: This is really hard to debug sometimes.
    """

    def the_decorator(cls: Type):
        methods = []

        # find all InheritDynamic properties in all bases
        for ccls in inspect.getmro(cls):
            for method_name, method in ccls.__dict__.items():
                if isinstance(method, InheritDynamic):
                    # found one
                    # do not replace if the main class already overrides it
                    if isinstance(cls.__dict__.get(method_name, method), InheritDynamic):
                        methods.append((method_name, method))

        if len(methods) == 0:
            cls_name = cls.__name__
            decorator_name = extends_dynamic.__name__
            prop_name = InheritDynamic.__name__
            raise ExtendsDynamicError(
                f"Class {cls_name} uses the @{decorator_name} decorator, but has no properties of type {prop_name}."
            )

        orig_init = cls.__init__

        def __init__(self, *kargs, **kwargs):
            orig_init(self, *kargs, **kwargs)

            the_origin = getattr(self, origin_property)
            for method_name, _ in methods:
                underlying_func = getattr(the_origin, method_name)
                if isinstance(underlying_func, MethodType):
                    underlying_func = underlying_func.__func__
                bound_func = MethodType(underlying_func, self)
                object.__setattr__(self, method_name, bound_func)

        if hasattr(cls, "__getattr__"):
            orig_getattr = cls.__getattr__
        else:
            orig_getattr = None

        def __getattr__(self, key: str):
            # try finding the attribute in the origin
            the_origin = getattr(self, origin_property)
            if hasattr(the_origin, key):
                return getattr(the_origin, key)

            # use the default getattr
            if orig_getattr is not None:
                return orig_getattr(self, key)

            raise KeyError(f"{self.__class__.__name__}: Failed to locate property {key}.")

        cls.__init__ = __init__
        cls.__getattr__ = __getattr__

        return cls

    return the_decorator


def addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


class OrderedSet(Generic[_T], MutableSet[_T]):
    # __contains__, __iter__, __len__, add(), and discard()
    __slots__ = ("__dict", "__rev")

    def __init__(self, vals: Iterable[_T] | None = None) -> None:
        if vals is None:
            vals = iter(())

        self.__rev = list(vals)
        self.__dict = OrderedDict(((v, i) for i, v in enumerate(self.__rev)))

    def __contains__(self, x: object) -> bool:
        return x in self.__dict

    def __iter__(self) -> Iterator[_T]:
        return iter(self.__dict)

    def __len__(self) -> int:
        return len(self.__dict)

    def add(self, value: _T) -> None:
        if value in self.__dict:
            return

        self.__dict[value] = len(self.__dict)
        self.__rev.append(value)

    def _reenumerate(self):
        for i, k in enumerate(self.__dict):
            self.__dict[k] = i

    def _discard(self, value: _T) -> bool:
        if value not in self.__dict:
            return False

        pos = self.__dict[value]
        del self.__dict[value]
        del self.__rev[pos]
        return True

    def discard(self, value: _T) -> None:
        if self._discard(value):
            self._reenumerate()

    def discard_all(self, *values: _T):
        if any((self._discard(v) for v in values)):
            self._reenumerate()

    def index_of(self, value: _T) -> int:
        return self.__dict[value]

    def __getitem__(self, i: int) -> _T:
        return self.__rev[i]
