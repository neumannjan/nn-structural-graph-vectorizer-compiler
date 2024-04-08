from typing import Collection, Iterator

from lib.vectorize.model.repr import repr_slots


class Refs(Collection[tuple[int, str, int]]):
    __slots__ = ("types", "layer_ids", "ordinals")
    __repr__ = repr_slots

    TYPE_FACT = 0
    TYPE_WEIGHT = 1
    TYPE_LAYER = 2

    def __init__(self, types: list[int], layer_ids: list[str], ordinals: list[int]) -> None:
        self.types = types
        self.layer_ids = layer_ids
        self.ordinals = ordinals

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


class LayerRefs(Collection[tuple[int, str]]):
    __slots__ = ("types", "layer_ids")
    __repr__ = repr_slots

    TYPE_FACT = 0
    TYPE_WEIGHT = 1
    TYPE_LAYER = 2

    def __init__(self, types: list[int], layer_ids: list[str]) -> None:
        self.types = types
        self.layer_ids = layer_ids

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
