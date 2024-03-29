import itertools
from dataclasses import fields, is_dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from typing_extensions import Callable

from lib.utils import addindent


def _get_key(key) -> str:
    if isinstance(key, str):
        return key
    else:
        return my_repr(key)


def _iter_items(obj: object):
    if hasattr(obj.__class__, "__slots__"):
        keys = obj.__class__.__slots__  # pyright: ignore
        yield from ((k, getattr(obj, k)) for k in keys)
    elif is_dataclass(obj):
        yield from ((k.name, getattr(obj, k.name)) for k in fields(obj))
    elif isinstance(obj, Mapping):
        yield from obj.items()
    else:
        assert False


def repr_module_like(self: object, is_module: Callable[[Any, Any], bool]) -> str:
    modules = dict(((k, v) for k, v in _iter_items(self) if is_module(k, v)))
    extras = dict(((k, v) for k, v in _iter_items(self) if not is_module(k, v)))

    extra_lines = []
    extra_repr = ", ".join((f"{_get_key(k)}={my_repr(v)}" for k, v in extras.items()))
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split("\n")
    child_lines = []
    for key, module in modules.items():
        mod_str = my_repr(module)
        mod_str = addindent(mod_str, 2)
        child_lines.append("(" + _get_key(key) + "): " + mod_str)
    lines = extra_lines + child_lines

    main_str = ""
    if lines:
        # simple one-liner info
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += "\n  " + "\n  ".join(lines) + "\n"

    return main_str


def my_repr(self) -> str:
    if isinstance(self, np.ndarray):
        return f"array({self.shape})"
    elif isinstance(self, str):
        return repr(self)
    elif (
        isinstance(self, object)
        and not isinstance(self, (list, dict, tuple))
        and hasattr(self.__class__, "__repr__")
        and self.__class__.__repr__ != my_repr
        and self.__class__.__repr__ != object.__repr__
    ):
        return repr(self)
    elif isinstance(self, Sequence):
        n = 3
        if len(self) <= n:
            return f"[{', '.join((my_repr(v) for v in self))}]"
        else:
            return f"[{', '.join((my_repr(v) for v in self[:n]))}, ... (size: {len(self)})]"
    elif isinstance(self, Mapping):
        n = 3
        vals = ", ".join((f"{_get_key(k)}={my_repr(v)}" for k, v in itertools.islice(self.items(), 3)))
        if len(self) <= n:
            return f"{{{vals}}}"
        else:
            return f"{{{vals}, ... (size: {len(vals)})}}"
    else:
        return repr(self)


def repr_slots(self: object) -> str:
    name = self.__class__.__name__
    return (
        name
        + "("
        + repr_module_like(self, is_module=lambda k, v: isinstance(v, object) and hasattr(v.__class__, "__slots__"))
        + ")"
    )


class ModuleDictWrapper:
    __slots__ = ("value",)

    def __init__(self, value: dict) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"{{{repr_module_like(self.value, is_module=lambda k, v: True)}}}"
