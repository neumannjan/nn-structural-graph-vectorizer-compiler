from typing import Callable, Generic, TypeVar

_T = TypeVar("_T")
_R = TypeVar("_R")


class Pipe(Generic[_T]):
    def __init__(self, value: _T) -> None:
        self._value = value

    def __add__(self, func: Callable[[_T], _R]) -> "Pipe[_R]":
        return Pipe(func(self._value))

    @property
    def value(self):
        return self._value
