import functools
import heapq
import re
from collections.abc import Collection, Mapping
from dataclasses import fields, is_dataclass
from typing import (
    Any,
    Callable,
    Container,
    Generic,
    Iterable,
    Iterator,
    Sequence,
    TypeVar,
    overload,
)

import numpy as np


def atleast_2d_shape(shape: Sequence[int]) -> Sequence[int]:
    dim = len(shape)

    if dim == 0:
        return [1, 1]
    elif dim == 1:
        return [shape[0], 1]
    else:
        return shape


def _argmax_last(a: np.ndarray):
    a = a[::-1]
    return len(a) - np.argmax(a) - 1


def _detect_possible_last_incomplete(inp: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    # find last occurrence of the initial string
    last_idx = _argmax_last((inp == inp[0]).reshape([inp.shape[0], -1]).all(axis=-1))

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


def detect_repeating_sequence_in_list(inp: Sequence | np.ndarray, allow_last_incomplete=False) -> int | None:
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
    4
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2], allow_last_incomplete=False)
    4
    >>> detect_repeating_sequence_in_list([0], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([0, 1], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([0, 1, 2], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0], allow_last_incomplete=True)
    4
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1], allow_last_incomplete=True)
    4
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1, 2], allow_last_incomplete=True)
    4
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2], allow_last_incomplete=True)
    4
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0], allow_last_incomplete=True)
    4
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1], allow_last_incomplete=True)
    4
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2], allow_last_incomplete=True)
    4
    >>> detect_repeating_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2], allow_last_incomplete=True)
    4
    >>> detect_repeating_sequence_in_list([0], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([0, 0], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([0, 0, 0], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([0], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([0, 0], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([0, 0, 0], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([2, 0], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([2, 0, 1], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2], allow_last_incomplete=False)
    4
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1], allow_last_incomplete=False)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2], allow_last_incomplete=False)
    4
    >>> detect_repeating_sequence_in_list([2, 0], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([2, 0, 1], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2], allow_last_incomplete=True)
    3
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2], allow_last_incomplete=True)
    4
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0], allow_last_incomplete=True)
    4
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1], allow_last_incomplete=True)
    4
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2], allow_last_incomplete=True)
    4
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2], allow_last_incomplete=True)
    4
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0], allow_last_incomplete=True)
    4
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1], allow_last_incomplete=True)
    4
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2], allow_last_incomplete=True)
    4
    >>> detect_repeating_sequence_in_list([2, 0, 1, 3], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 3], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 3], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 3], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 3], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 3], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 3], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 3], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 3], allow_last_incomplete=True)
    >>> detect_repeating_sequence_in_list([['1', '14', 0], ['1', '15', 0], ['0', 'unit', '0'],
    ... ['1', '14', 0], ['1', '15', 0], ['0', 'unit', '0']])
    3
    >>> detect_repeating_sequence_in_list([['1', '14', 0], ['1', '15', 0], ['0', 'unit', '0'],
    ... ['1', '14', 0], ['1', '15', 0]], allow_last_incomplete=True)
    3
    """
    inp = np.array(inp)

    # we need to find possible period lengths (K):

    # find only Ks up to the middle of the sequence (because the string is repeated at least twice), which are also
    # equal to the initial value
    mask = inp[: len(inp) // 2 + 1] == inp[0]
    mask = mask.reshape([mask.shape[0], -1]).all(axis=-1)

    (candidate_Ks,) = np.where(mask)

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
            mask = inp_first[: len(inp_first) // 2 + 1] == inp_first[0]
            mask = mask.reshape([mask.shape[0], -1]).all(axis=-1)

            (candidate_Ks,) = np.where(mask)

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

        if np.all(inp_first.reshape([-1, k, *inp_first.shape[1:]]) == inp_first[:k]) and (
            inp_last is None or np.all(inp_last == inp_first[: len(inp_last)])
        ):
            if k == 1:
                return None
            return k

    return None


def detect_repeating_K_sequence_in_list(
    inp: Sequence | np.ndarray, period: int, allow_last_incomplete=False
) -> int | None:
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
    4
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2], 4, allow_last_incomplete=False)
    4
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
    4
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1], 4, allow_last_incomplete=True)
    4
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2], 4, allow_last_incomplete=True)
    4
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2], 4, allow_last_incomplete=True)
    4
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0], 4, allow_last_incomplete=True)
    4
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1], 4, allow_last_incomplete=True)
    4
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2], 4, allow_last_incomplete=True)
    4
    >>> detect_repeating_K_sequence_in_list([0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2], 4, allow_last_incomplete=True)
    4
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
    3
    >>> detect_repeating_K_sequence_in_list([0, 0, 0, 0, 0, 0, 0], 3, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([0], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([0, 0], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([0, 0, 0], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([0, 0, 0, 0], 3, allow_last_incomplete=True)
    3
    >>> detect_repeating_K_sequence_in_list([0, 0, 0, 0, 0], 3, allow_last_incomplete=True)
    3
    >>> detect_repeating_K_sequence_in_list([0, 0, 0, 0, 0, 0], 3, allow_last_incomplete=True)
    3
    >>> detect_repeating_K_sequence_in_list([0, 0, 0, 0, 0, 0, 0], 3, allow_last_incomplete=True)
    3
    >>> detect_repeating_K_sequence_in_list([2, 0], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2], 4, allow_last_incomplete=False)
    4
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1], 4, allow_last_incomplete=False)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2], 4, allow_last_incomplete=False)
    4
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
    4
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0], 4, allow_last_incomplete=True)
    4
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1], 4, allow_last_incomplete=True)
    4
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2], 4, allow_last_incomplete=True)
    4
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2], 4, allow_last_incomplete=True)
    4
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0], 4, allow_last_incomplete=True)
    4
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1], 4, allow_last_incomplete=True)
    4
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2], 4, allow_last_incomplete=True)
    4
    >>> detect_repeating_K_sequence_in_list([2, 0], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1], 3, allow_last_incomplete=True)
    >>> detect_repeating_K_sequence_in_list([2, 0, 1, 2], 3, allow_last_incomplete=True)
    3
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
    3
    >>> detect_repeating_K_sequence_in_list([['1', '14', 0], ['1', '15', 0], ['0', 'unit', '0'],
    ... ['1', '14', 0], ['1', '15', 0]], 3, allow_last_incomplete=True)
    3
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
        return k

    return None


def detect_repeating_interleaved_sequence_in_list(
    inp: Sequence | np.ndarray, allow_last_incomplete=False
) -> int | None:
    """
    TODO doc.

    >>> detect_repeating_interleaved_sequence_in_list([0])
    >>> detect_repeating_interleaved_sequence_in_list([0, 1])
    >>> detect_repeating_interleaved_sequence_in_list([0, 0])
    2
    >>> detect_repeating_interleaved_sequence_in_list([0, 0, 1, 1])
    2
    >>> detect_repeating_interleaved_sequence_in_list([0, 0, 0, 1, 1, 1])
    3
    >>> detect_repeating_interleaved_sequence_in_list([0, 1, 2])
    >>> detect_repeating_interleaved_sequence_in_list([0, 0, 1, 1, 2, 2])
    2
    >>> detect_repeating_interleaved_sequence_in_list([0, 0, 0, 1, 1, 1, 2, 2, 2])
    3
    >>> detect_repeating_interleaved_sequence_in_list([0, 0, 0, 1, 1, 1, 2, 2])
    >>> detect_repeating_interleaved_sequence_in_list([0, 0, 0, 1, 1, 1, 2, 2], allow_last_incomplete=True)
    3
    >>> detect_repeating_interleaved_sequence_in_list([0, 0, 0, 1, 1, 2, 2, 2])
    >>> detect_repeating_interleaved_sequence_in_list([0, 0, 0, 1, 1, 2, 2, 2], allow_last_incomplete=True)
    >>> detect_repeating_interleaved_sequence_in_list([['1', '14', 0], ['1', '15', 0]])
    >>> detect_repeating_interleaved_sequence_in_list([['1', '14', 0], ['1', '14', 0], ['1', '15', 0], ['1', '15', 0]])
    2
    >>> detect_repeating_interleaved_sequence_in_list([['1', '14', 0], ['1', '14', 0],
    ... ['1', '15', 0], ['1', '15', 0], ['1', '16', 0]])
    >>> detect_repeating_interleaved_sequence_in_list([['1', '14', 0], ['1', '14', 0],
    ... ['1', '15', 0], ['1', '15', 0], ['1', '16', 0]], allow_last_incomplete=True)
    2
    """
    if not isinstance(inp, np.ndarray):
        inp = np.array(inp)

    repeats_mask = (inp[0] == inp).reshape([inp.shape[0], -1]).all(axis=-1)
    n_period = int(np.argmin(repeats_mask))
    if n_period == 0 and repeats_mask[n_period]:
        # all are the same value
        n_period = inp.shape[0]
    elif repeats_mask[n_period]:
        return None

    if n_period <= 1:
        return None

    if not allow_last_incomplete and (inp.shape[0] % n_period) != 0:
        return None

    expected = np.repeat(inp[::n_period], n_period, axis=0)[: len(inp)]

    if expected.shape == inp.shape and np.all(expected == inp):
        return n_period

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


_CAMEL_TO_SNAKE_REGEX1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_TO_SNAKE_REGEX2 = re.compile(r"([a-z0-9])([A-Z])")


def camel_to_snake(name: str, upper=False):
    name = _CAMEL_TO_SNAKE_REGEX1.sub(r"\1_\2", name)
    name = _CAMEL_TO_SNAKE_REGEX2.sub(r"\1_\2", name)
    out = name.upper() if upper else name.lower()
    return out


class AnyWhitelist(Container[_T], Generic[_T]):
    def __contains__(self, x: object, /) -> bool:
        return True


class Blacklist(Container[_T], Generic[_T]):
    def __init__(self, values: Container[_T]) -> None:
        self.values = values

    def __contains__(self, x: object, /) -> bool:
        return x not in self.values


_KEY_TO_ABBR_REGEX = re.compile("(?:_|^)(\\w)")


def key_to_abbr(key: str) -> str:
    out = _KEY_TO_ABBR_REGEX.findall(key)
    out = "".join(out)
    return out


def _dataclass_to_shorthand(o: object, prefix: str = ""):
    assert is_dataclass(o) and isinstance(o, object)

    for field in fields(o):
        k = prefix + key_to_abbr(field.name)
        v = getattr(o, field.name)

        if is_dataclass(v) and isinstance(v, object):
            yield f"{k}=[{dataclass_to_shorthand(v)}]"
        else:
            yield f"{k}={v}"


def dataclass_to_shorthand(o: object, prefix: str = ""):
    return ",".join(_dataclass_to_shorthand(o, prefix))


def iter_empty(it: Iterator) -> bool:
    try:
        next(it)
    except StopIteration:
        return True
    return False


def serialize_dataclass(self: object, call_self=True) -> dict[str, Any]:
    assert is_dataclass(self) and isinstance(self, object)

    if call_self and "serialize" in self.__class__.__dict__:
        return self.serialize()  # pyright: ignore

    out = {}

    for field in fields(self):
        k = field.name
        v = getattr(self, field.name)

        if is_dataclass(v):
            out[k] = serialize_dataclass(v)
        else:
            out[k] = v

    return out
