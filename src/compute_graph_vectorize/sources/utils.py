from typing import Iterable, Iterator, OrderedDict

from compute_graph_vectorize.sources.base import LayerDefinition
from compute_graph_vectorize.sources.base_impl import BaseLayerDefinitions


class LayerDefinitionsImpl(BaseLayerDefinitions):
    def __init__(self, data: OrderedDict[str, LayerDefinition]) -> None:
        self._data = data

    @classmethod
    def from_iter(cls, layer_definitions: Iterable[LayerDefinition]):
        data = OrderedDict(((ld.id, ld) for ld in layer_definitions))
        return cls(data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, layer_id: str) -> LayerDefinition:
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
