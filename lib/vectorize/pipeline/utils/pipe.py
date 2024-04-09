from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar, overload

_T = TypeVar("_T")
_R = TypeVar("_R")
_S = TypeVar("_S")


class OpPipe(ABC, Generic[_T, _R]):
    @abstractmethod
    def __call__(self, input: _T) -> _R: ...

    @abstractmethod
    def __add__(self, func: Callable[[_R], _S]) -> "OpPipe[_T, _S]": ...


class _OpsSequence(OpPipe[_T, _R], Generic[_T, _R]):
    def __init__(self, ops: list[Callable[[Any], Any]]) -> None:
        self._ops = ops

    def __call__(self, input: _T) -> _R:
        x = input
        for op in self._ops:
            x = op(x)

        return x  # pyright: ignore

    def __add__(self, func: Callable[[_R], _S]) -> "OpPipe[_T, _S]":
        ops2: list[Callable[[Any], Any]] = []
        ops2.extend(self._ops)

        if isinstance(func, _OpsSequence):
            ops2.extend(func._ops)
        else:
            ops2.append(func)

        return _OpsSequence(ops2)

    def __iadd__(self, func: Callable[[_R], _S]) -> "OpPipe[_T, _S]":
        if isinstance(func, _OpsSequence):
            self._ops.extend(func._ops)
        else:
            self._ops.append(func)
        return self  # pyright: ignore


class IdentityPipe:
    def __call__(self, input: _T) -> _T:
        return input

    def __add__(self, func: Callable[[_T], _R]) -> OpPipe[_T, _R]:
        if isinstance(func, _OpsSequence):
            return func
        else:
            return _OpsSequence([func])


PIPE = IdentityPipe()
