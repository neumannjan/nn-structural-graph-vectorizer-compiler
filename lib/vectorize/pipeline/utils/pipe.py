from typing import Callable, Generic, TypeVar

_T = TypeVar("_T")
_R = TypeVar("_R")
_S = TypeVar("_S")


class Pipe:
    def __call__(self, input: _T) -> _T:
        return input

    def __add__(self, func: Callable[[_T], _R]) -> "OpPipe[_T, _R]":
        return OpPipe(func)


class OpPipe(Generic[_T, _R]):
    def __init__(self, func: Callable[[_T], _R]) -> None:
        self._func = func

    def __call__(self, input: _T) -> _R:
        return self._func(input)

    def __add__(self, func: Callable[[_R], _S]) -> "OpPipe[_T, _S]":
        return OpPipe(lambda t: func(self._func(t)))


PIPE = Pipe()
