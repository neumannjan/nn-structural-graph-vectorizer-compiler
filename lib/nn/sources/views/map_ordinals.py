from typing import Mapping

from lib.nn.sources.base import LayerOrdinal, Network
from lib.nn.sources.minimal_api.ordinals import MinimalAPIOrdinals
from lib.nn.sources.views.view import OrdinalsViewBasis, View, ordinals_view_basis, view
from lib.utils import MapMapping


@ordinals_view_basis
class MapOrdinalsViewBasis(OrdinalsViewBasis):
    def __init__(self, ordinals: MinimalAPIOrdinals, ordinals_map: dict[LayerOrdinal, LayerOrdinal]) -> None:
        super().__init__(ordinals)
        self.__ordinals_map = ordinals_map

    def __map_ordinal(self, o: LayerOrdinal):
        o = self.__ordinals_map.get(o, o)
        return o

    def get_ordinals_for_layer(self, layer_id: int) -> Mapping[int, LayerOrdinal]:
        ords = self.ordinals.get_ordinals_for_layer(layer_id)
        return MapMapping(self.__map_ordinal, ords)


@view(MapOrdinalsViewBasis)
class MapOrdinalsView(View):
    def __new__(cls, network: Network, ordinals_map: dict[LayerOrdinal, LayerOrdinal]):
        if len(ordinals_map) == 0:
            return network

        return super().__new__(cls)


    def __init__(self, network: Network, ordinals_map: dict[LayerOrdinal, LayerOrdinal]) -> None: ...
