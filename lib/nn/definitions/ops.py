from typing import Literal

# TODO: support missing transformations
TransformationDef = Literal[
    "identity",
    "tanh",
    "square_root",
    "signum",
    "sign",
    "logarithm",
    "log",
    "ln",
    "relu",
    "leaky_relu",
    "exponentiation",
    "exp",
    "sigmoid",
]
