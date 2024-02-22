import itertools
import os

from lib.nn.topological.settings import Settings
from torch.backends.mps import is_available as is_mps_available
from torch.cuda import is_available as is_cuda_available

SETTINGS_PARAMS = [
    Settings(
        check_same_layers_assumption=check_same_layers_assumption,
    )
    for (check_same_layers_assumption, optimize_linear_gathers) in itertools.product(  # -
        [False],  # check_same_layers_assumption
        [True, False],  # optimize_linear_gathers
    )
]

DEVICE_PARAMS = [
    "cpu",
]

CPU_ONLY = os.getenv("CPU_ONLY", "false").lower() in ("t", "true", "y", "yes", "on", "1")

if not CPU_ONLY:
    if is_cuda_available():
        DEVICE_PARAMS += [
            "cuda",
        ]

    if is_mps_available():
        DEVICE_PARAMS += [
            "mps",
        ]
