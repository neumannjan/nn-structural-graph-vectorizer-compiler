from typing import Collection, Iterable, Iterator, OrderedDict

from lib.nn.sources.base_source import BaseLayerDefinitions, BaseOrdinals
from lib.nn.sources.source import LayerDefinition, LayerOrdinal, Ordinals
from lib.utils import cache


class OrdinalsDict(BaseOrdinals):
    def __init__(self, data: dict[int, LayerOrdinal]) -> None:
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def ids(self) -> Collection[int]:
        return self._data.keys()

    def items(self) -> Collection[tuple[int, LayerOrdinal]]:
        return self._data.items()

    def values(self) -> Collection[LayerOrdinal]:
        return self._data.values()

    def __iter__(self) -> Iterator[LayerOrdinal]:
        return iter(self._data.values())

    @property
    @cache
    def _values_set(self) -> set[LayerOrdinal]:
        return set(self._data.values())

    def __contains__(self, o: object) -> bool:
        if isinstance(o, LayerOrdinal):
            return o in self._values_set

        if isinstance(o, int):
            return o in self._data

        raise ValueError(f"Cannot check if value is in ordinals for value {o} of type {type(o)}.")

    def __getitem__(self, id: int) -> LayerOrdinal:
        return self._data[id]


class LayerDefinitionsImpl(BaseLayerDefinitions):
    def __init__(self, data: OrderedDict[int, LayerDefinition]) -> None:
        self._data = data

    @classmethod
    def from_iter(cls, layer_definitions: Iterable[LayerDefinition]):
        data = OrderedDict(((ld.id, ld) for ld in layer_definitions))
        return cls(data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, layer_id: int) -> LayerDefinition:
        return self._data[layer_id]

    def __iter__(self) -> Iterator[LayerDefinition]:
        return iter(self._data.values())

    def __contains__(self, ld: object) -> bool:
        if isinstance(ld, LayerDefinition):
            return ld.id in self._data

        return ld in self._data

    def as_list(self) -> list[LayerDefinition]:
        return list(self._data.values())

    def as_dict(self) -> OrderedDict[int, LayerDefinition]:
        return OrderedDict(self._data)


class OrdinalsProxy(BaseOrdinals):
    def __init__(
        self,
        network_ordinals: Ordinals,
        ids: Collection[int],
        items: Collection[tuple[int, LayerOrdinal]],
        values: Collection[LayerOrdinal],
    ) -> None:
        self._ids = ids
        self._items = items
        self._values = values
        self._all_ordinals = network_ordinals

    def ids(self) -> Collection[int]:
        return self._ids

    def items(self) -> Collection[tuple[int, LayerOrdinal]]:
        return self._items

    def values(self) -> Collection[LayerOrdinal]:
        return self._values

    def __iter__(self) -> Iterator[LayerOrdinal]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    @property
    @cache
    def _ids_set(self) -> set[int]:
        return set(self._ids)

    @property
    @cache
    def _values_set(self) -> set[LayerOrdinal]:
        return set(self._values)

    def __contains__(self, o: object) -> bool:
        if isinstance(o, LayerOrdinal):
            return o in self._values_set

        if isinstance(o, int):
            return o in self._ids_set

        raise ValueError(f"Cannot check if value is in ordinals for value {o} of type {type(o)}.")

    def __getitem__(self, id: int) -> LayerOrdinal:
        if id in self._ids_set:
            return self._all_ordinals[id]

        raise KeyError(id)
