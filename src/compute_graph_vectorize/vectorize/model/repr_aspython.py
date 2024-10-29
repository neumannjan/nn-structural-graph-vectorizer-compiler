from collections import OrderedDict
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from compute_graph_vectorize.utils import addindent


@runtime_checkable
class PrintableAsPython(Protocol):
    def repr_as_python(self) -> str: ...


def _get_value(obj: object, key: str | int):
    if isinstance(obj, Mapping):
        return obj[key]

    if isinstance(key, int):
        if hasattr(obj.__class__, "__getitem__"):
            return obj[key]  # pyright: ignore

    return getattr(obj, str(key))


def _iter_values(obj: object, keys: Iterable[str | int]):
    yield from (_get_value(obj, key) for key in keys)


def _prepr_pair_object(key, value: str):
    return f"{key}={value}"


def _prepr_pair_dict(key, value: str):
    return f"{prepr(key)}: {value}"


def _prepr_pair_ordereddict(key, value: str):
    return f"({prepr(key)}, {value})"


def prepr_module_like(
    self: object,
    module_keys: Iterable[str | int],
    extra_keys: Iterable[str | int],
    pair: Callable[[Any, str], str] = _prepr_pair_object,
) -> str:
    modules = dict(((k, v) for k, v in zip(module_keys, _iter_values(self, module_keys))))
    extras = dict(((k, v) for k, v in zip(extra_keys, _iter_values(self, extra_keys))))

    extra_lines = []
    extra_repr = ", ".join((pair(k, prepr(v)) for k, v in extras.items()))
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split("\n")
    child_lines = []
    for key, module in modules.items():
        mod_str = prepr(module)
        mod_str = addindent(mod_str, 2)
        child_lines.append(pair(key, mod_str) + ",")

    if extra_repr and child_lines:
        extra_lines[-1] += ","

    lines = extra_lines + child_lines

    main_str = ""
    if lines:
        # simple one-liner info
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += "\n  " + "\n  ".join(lines) + "\n"

    return main_str


def prepr(self) -> str:
    if isinstance(self, np.ndarray):
        if self.size > 20:
            return f"np.random.random_sample({self.shape})"
        else:
            return f"np.array({repr(list(self.flatten()))}).reshape({repr(list(self.shape))})"
    elif isinstance(self, str):
        return '"' + self.replace('"', '\\"') + '"'
    elif isinstance(self, PrintableAsPython):
        return self.repr_as_python()
    elif isinstance(self, object) and hasattr(self.__class__, "__slots__"):
        return prepr_slots(self)
    elif isinstance(self, Sequence):
        return f"[{', '.join((prepr(v) for v in self))}]"
    elif isinstance(self, OrderedDict):
        return f"OrderedDict([{prepr_module_like(self, module_keys=self.keys(), extra_keys=[], pair=_prepr_pair_ordereddict)}])"
    elif isinstance(self, Mapping):
        return f"{{{prepr_module_like(self, module_keys=self.keys(), extra_keys=[], pair=_prepr_pair_dict)}}}"
    else:
        return repr(self)


def _attr_has_slots(obj: object, attr: str):
    value = getattr(obj, attr)

    return isinstance(value, object) and hasattr(value.__class__, "__slots__")


def prepr_slots(self: object) -> str:
    name = self.__class__.__name__
    assert hasattr(self.__class__, "__slots__")
    return (
        name
        + "("
        + prepr_module_like(
            self,
            module_keys=[s for s in self.__class__.__slots__ if _attr_has_slots(self, s)],  # pyright: ignore
            extra_keys=[s for s in self.__class__.__slots__ if not _attr_has_slots(self, s)],  # pyright: ignore
        )
        + ")"
    )


class ModuleDictWrapper:
    __slots__ = ("value",)

    def __init__(self, value: dict) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"{{{prepr_module_like(self.value, module_keys=self.value.keys(), extra_keys=())}}}"
