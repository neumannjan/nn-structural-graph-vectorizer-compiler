from dataclasses import dataclass
from typing import Literal

Compilation = Literal["none", "trace", "script"]
TorchReduceMethod = Literal["segment_csr", "scatter"]


@dataclass
class TorchModuleSettings:
    reduce_method: TorchReduceMethod = "segment_csr"

    compilation: Compilation = "none"
