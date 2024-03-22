import os

from lib.nn.definitions.settings import Settings
from torch.backends.mps import is_available as is_mps_available
from torch.cuda import is_available as is_cuda_available

D = Settings()


# assert that it matches what I think it does
# allows me to immediately see where I'm using the default values, as well as what the values are
def _D(a, b):
    assert a == b
    return a


SETTINGS_PARAMS = [
    Settings(
        optimize_linear_gathers=_D(D.optimize_linear_gathers, True),
        group_learnable_weight_parameters=_D(D.group_learnable_weight_parameters, True),
        allow_non_builtin_torch_ops=_D(D.allow_non_builtin_torch_ops, True),
        optimize_tail_gathers=_D(D.optimize_tail_gathers, True),
        use_unique_pre_gathers=_D(D.use_unique_pre_gathers, False),
    ),
    Settings(
        optimize_linear_gathers=not _D(D.optimize_linear_gathers, True),
        group_learnable_weight_parameters=_D(D.group_learnable_weight_parameters, True),
        allow_non_builtin_torch_ops=_D(D.allow_non_builtin_torch_ops, True),
        optimize_tail_gathers=not _D(D.optimize_tail_gathers, True),
        use_unique_pre_gathers=_D(D.use_unique_pre_gathers, False),
    ),
    Settings(
        optimize_linear_gathers=_D(D.optimize_linear_gathers, True),
        group_learnable_weight_parameters=not _D(D.group_learnable_weight_parameters, True),
        allow_non_builtin_torch_ops=_D(D.allow_non_builtin_torch_ops, True),
        optimize_tail_gathers=not _D(D.optimize_tail_gathers, True),
        use_unique_pre_gathers=_D(D.use_unique_pre_gathers, False),
    ),
    Settings(
        optimize_linear_gathers=_D(D.optimize_linear_gathers, True),
        group_learnable_weight_parameters=_D(D.group_learnable_weight_parameters, True),
        allow_non_builtin_torch_ops=not _D(D.allow_non_builtin_torch_ops, True),
        optimize_tail_gathers=not _D(D.optimize_tail_gathers, True),
        use_unique_pre_gathers=_D(D.use_unique_pre_gathers, False),
    ),
    Settings(
        optimize_linear_gathers=_D(D.optimize_linear_gathers, True),
        group_learnable_weight_parameters=_D(D.group_learnable_weight_parameters, True),
        allow_non_builtin_torch_ops=_D(D.allow_non_builtin_torch_ops, True),
        optimize_tail_gathers=not _D(D.optimize_tail_gathers, True),
        use_unique_pre_gathers=_D(D.use_unique_pre_gathers, False),
    ),
    Settings(
        optimize_linear_gathers=_D(D.optimize_linear_gathers, True),
        group_learnable_weight_parameters=_D(D.group_learnable_weight_parameters, True),
        allow_non_builtin_torch_ops=_D(D.allow_non_builtin_torch_ops, True),
        optimize_tail_gathers=not _D(D.optimize_tail_gathers, True),
        use_unique_pre_gathers=not _D(D.use_unique_pre_gathers, False),
    ),
    Settings(
        optimize_linear_gathers=_D(D.optimize_linear_gathers, True),
        group_learnable_weight_parameters=_D(D.group_learnable_weight_parameters, True),
        allow_non_builtin_torch_ops=_D(D.allow_non_builtin_torch_ops, True),
        optimize_tail_gathers=_D(D.optimize_tail_gathers, True),
        use_unique_pre_gathers=not _D(D.use_unique_pre_gathers, False),
    ),
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
