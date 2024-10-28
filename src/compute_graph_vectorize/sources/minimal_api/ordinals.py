from collections.abc import Mapping
from typing import OrderedDict, Protocol, overload

from compute_graph_vectorize.sources.base import LayerOrdinal
from compute_graph_vectorize.sources.minimal_api.base import MinimalAPINetwork
from compute_graph_vectorize.utils import cache


class MinimalAPIOrdinals(Protocol):
    """
    Minimal (i.e. easy to implement) API for a neural network ordinals provider.

    Provides mappings between neuron ids and ordinals.

    For convenience, the end user is advised to use the full API, for which there exists a universal implementation
    for an arbitrary `MinimalAPINetwork` and `MinimalAPIOrdinals`.

    See documentation for `MinimalAPINetwork` for more info.
    """

    @overload
    def get_ordinal(self, layer: str | None, id: int, /) -> LayerOrdinal: ...

    @overload
    def get_ordinal(self, id: int, /) -> LayerOrdinal: ...

    def get_ordinal(self, *kargs: str | int | None) -> LayerOrdinal: ...

    def get_id(self, o: LayerOrdinal) -> int: ...

    def get_ordinals_for_layer(self, layer_id: int) -> Mapping[int, LayerOrdinal]: ...

    def get_all_ordinals(self) -> dict[int, LayerOrdinal]: ...


def get_ordinal_args(*kargs: str | int | None) -> tuple[str | None, int]:
    if len(kargs) == 1:
        layer = None
        (id,) = kargs
        assert isinstance(id, int)
    elif len(kargs) == 2:
        layer, id = kargs
        assert layer is None or isinstance(layer, str)
        assert isinstance(id, int)
    else:
        raise RuntimeError()
    assert id is not None
    return layer, id


class MinimalAPIOrdinalsImpl(MinimalAPIOrdinals):
    """`MinimalAPIOrdinals` implementation where ordinals are calculated per layer based on a `MinimalAPINetwork`."""

    def __init__(self, minimal_api: MinimalAPINetwork) -> None:
        self._minimal_api = minimal_api

    @overload
    def get_ordinal(self, layer: str | None, id: int, /) -> LayerOrdinal: ...

    @overload
    def get_ordinal(self, id: int, /) -> LayerOrdinal: ...

    @cache
    def get_ordinals_for_layer(self, layer_id: str) -> Mapping[int, LayerOrdinal]:
        neurons = self._minimal_api.get_layer_neurons(layer_id)
        ids = self._minimal_api.get_ids(neurons)
        return OrderedDict(((id, LayerOrdinal(layer=layer_id, ordinal=o)) for o, id in enumerate(ids)))

    @cache
    def get_all_ordinals(self) -> dict[int, LayerOrdinal]:
        out: dict[int, LayerOrdinal] = {}

        for layer in self._minimal_api.get_layers():
            out.update(self.get_ordinals_for_layer(layer.id).items())

        return out

    def get_ordinal(self, *kargs: str | int | None) -> LayerOrdinal:
        layer, id = get_ordinal_args(*kargs)

        if layer is not None:
            return self.get_ordinals_for_layer(layer)[id]
        else:
            return self.get_all_ordinals()[id]

    @cache
    def _reverse_ordinals_for_layer(self, layer_id: str) -> dict[LayerOrdinal, int]:
        ofl = self.get_ordinals_for_layer(layer_id)
        return {o: id for id, o in ofl.items()}

    def get_id(self, o: LayerOrdinal) -> int:
        rev = self._reverse_ordinals_for_layer(o.layer)
        return rev[o]
