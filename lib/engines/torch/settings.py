from dataclasses import dataclass
from typing import Literal

Compilation = Literal["none", "trace", "script"]


@dataclass
class TorchModuleSettings:
    allow_non_builtin_torch_ops: bool = True

    compilation: Compilation = "none"
