from collections.abc import Sequence
from typing import Iterable, Iterator, overload

from compute_graph_vectorize.vectorize.model.repr import repr_slots


def _type_to_str(type: int):
    if type == Refs.TYPE_FACT:
        return "f"
    elif type == Refs.TYPE_WEIGHT:
        return "w"
    elif type == Refs.TYPE_LAYER:
        return "l"
    else:
        return str(type)


class Refs(Sequence[tuple[int, str, int]]):
    __slots__ = ("types", "layer_ids", "ordinals")

    TYPE_FACT = 0
    TYPE_WEIGHT = 1
    TYPE_LAYER = 2

    def __init__(self, types: list[int], layer_ids: list[str], ordinals: list[int]) -> None:
        self.types = types
        self.layer_ids = layer_ids
        self.ordinals = ordinals

    @staticmethod
    def from_iter(it: Iterable[tuple[int, str, int]]):
        out = Refs([], [], [])
        for t, l, o in it:
            out.types.append(t)
            out.layer_ids.append(l)
            out.ordinals.append(o)
        return out

    def __contains__(self, x: object, /) -> bool:
        if not isinstance(x, tuple):
            return False

        if len(x) != 3:
            return False

        return x in zip(self.types, self.layer_ids, self.ordinals)

    def __iter__(self) -> Iterator[tuple[int, str, int]]:
        return zip(self.types, self.layer_ids, self.ordinals)

    def __len__(self) -> int:
        return len(self.types)

    @overload
    def __getitem__(self, key: slice) -> "Refs": ...

    @overload
    def __getitem__(self, key: int) -> tuple[int, str, int]: ...

    def __getitem__(self, key: slice | int) -> "Refs | tuple[int, str, int]":
        if isinstance(key, slice):
            return Refs(self.types[key], self.layer_ids[key], self.ordinals[key])

        return self.types[key], self.layer_ids[key], self.ordinals[key]

    def __setitem__(self, key: int, value: tuple[int, str, int]):
        self.types[key], self.layer_ids[key], self.ordinals[key] = value

    def set(self, values: Iterable[tuple[int, str, int]]):
        self.types = [r[0] for r in values]
        self.layer_ids = [r[1] for r in values]
        self.ordinals = [r[2] for r in values]

    def append(self, value: tuple[int, str, int]):
        t, l, o = value
        self.types.append(t)
        self.layer_ids.append(l)
        self.ordinals.append(o)

    def __repr__(self) -> str:
        n = 5
        out = ", ".join((f"<{_type_to_str(t)}|{l}|{o}>" for t, l, o in self[:n]))
        if len(self) > n:
            out += f", ... (size: {len(self)})"

        return f"{self.__class__.__name__}({out})"

    def __hash__(self) -> int:
        return hash((self.types, self.layer_ids, self.ordinals))

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, Refs)
            and len(self) == len(value)
            and all((a == b for a, b in zip(self.types, value.types)))
            and all((a == b for a, b in zip(self.layer_ids, value.layer_ids)))
            and all((a == b for a, b in zip(self.ordinals, value.ordinals)))
        )


class LayerRefs(Sequence[tuple[int, str]]):
    __slots__ = ("types", "layer_ids")
    __repr__ = repr_slots

    TYPE_FACT = 0
    TYPE_WEIGHT = 1
    TYPE_LAYER = 2

    def __init__(self, types: list[int], layer_ids: list[str]) -> None:
        self.types = types
        self.layer_ids = layer_ids

    @staticmethod
    def from_iter(it: Iterable[tuple[int, str]]):
        types = [t for t, _ in it]
        layer_ids = [l for _, l in it]
        return LayerRefs(types, layer_ids)

    def __contains__(self, x: object, /) -> bool:
        if not isinstance(x, tuple):
            return False

        if len(x) != 2:
            return False

        return x in zip(self.types, self.layer_ids)

    def __iter__(self) -> Iterator[tuple[int, str]]:
        return zip(self.types, self.layer_ids)

    def __len__(self) -> int:
        return len(self.types)

    @overload
    def __getitem__(self, key: slice) -> "LayerRefs": ...

    @overload
    def __getitem__(self, key: int) -> tuple[int, str]: ...

    def __getitem__(self, key: slice | int) -> "LayerRefs | tuple[int, str]":
        if isinstance(key, slice):
            return LayerRefs(self.types[key], self.layer_ids[key])

        return self.types[key], self.layer_ids[key]

    def __repr__(self) -> str:
        n = 5
        out = ", ".join((f"<{_type_to_str(t)}|{l}>" for t, l in self[:n]))
        if len(self) > n:
            out += f", ... (size: {len(self)})"

        return f"{self.__class__.__name__}({out})"

    def __hash__(self) -> int:
        return hash((tuple(self.types), tuple(self.layer_ids)))

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, LayerRefs)
            and len(self) == len(value)
            and all((a == b for a, b in zip(self.types, value.types)))
            and all((a == b for a, b in zip(self.layer_ids, value.layer_ids)))
        )
