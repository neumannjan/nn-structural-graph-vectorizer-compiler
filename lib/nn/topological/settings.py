from dataclasses import dataclass, field
from typing import Literal

from neuralogic.core import settings as nsettings

Compilation = Literal["none", "trace", "script"]


@dataclass
class Settings:
    merge_same_facts: bool = True

    # TODO: ASSUMPTION: all samples have the same layer layout
    check_same_layers_assumption: bool = False

    optimize_linear_gathers: bool = True

    # TODO: must be disabled (?) when batch is smaller than full (due to weights sharing)
    group_learnable_weight_parameters: bool = True

    allow_non_builtin_torch_ops: bool = True

    optimize_tail_gathers: bool = True

    compilation: Compilation = "none"
    neuralogic: nsettings.Settings = field(
        default_factory=lambda: nsettings.Settings(compute_neuron_layer_indices=True)
    )
