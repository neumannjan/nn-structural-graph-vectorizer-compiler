from typing import Protocol

from lib.nn.sources.minimal_api.base import MinimalAPINetwork
from lib.nn.sources.minimal_api.ordinals import MinimalAPIOrdinals


class MinimalAPINetworkAndOrdinals(MinimalAPINetwork, MinimalAPIOrdinals, Protocol): ...
