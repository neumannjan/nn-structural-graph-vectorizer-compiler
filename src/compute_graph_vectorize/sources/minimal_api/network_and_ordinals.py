from typing import Protocol

from compute_graph_vectorize.sources.minimal_api.base import MinimalAPINetwork
from compute_graph_vectorize.sources.minimal_api.ordinals import MinimalAPIOrdinals


class MinimalAPINetworkAndOrdinals(MinimalAPINetwork, MinimalAPIOrdinals, Protocol): ...
