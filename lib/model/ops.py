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

# TODO support missing aggregations
AggregationDef = Literal[
    "mean",
    "avg",
    "average",
    "softmax",
    "concat",
    "concatenation",
    "minimum",
    "min",
    "sum",
    "maximum",
    "max",
    "count",
]

ReductionDef = Literal[
    "mean",
    "avg",
    "average",
    "minimum",
    "min",
    "sum",
    "max",
    "maximum",
]
