from typing import Protocol

from lib.sources.minimal_api.base import MinimalAPINetwork
from lib.sources.minimal_api.ordinals import MinimalAPIOrdinals


class MinimalAPINetworkAndOrdinals(MinimalAPINetwork, MinimalAPIOrdinals, Protocol): ...
