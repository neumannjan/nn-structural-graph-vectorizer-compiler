import itertools
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pytest
import torch
from lib.nn import sources
from lib.nn.sources.base import LayerDefinition, Network
from lib.nn.sources.minimal_api.dict import WeightDefinitionImpl
from lib.nn.topological.linear import build_optimal_linear
from lib.tests.utils.neuron_factory import NeuronTestFactory
from lib.utils import atleast_3d_rev

LAYERS = [
    LayerDefinition(16, "FactLayer"),
    LayerDefinition(13, "WeightedRuleLayer"),
]


ValuesType = float | Literal["random"]


def build_value(t: ValuesType, *features):
    if t == "random":
        return np.random.random(features)
    else:
        return np.zeros(features) + t


def build_weight(id: int, t: ValuesType, in_features: int, out_features: int):
    val = build_value(t, out_features, in_features)
    return WeightDefinitionImpl(id=id, value=torch.tensor(val, dtype=torch.get_default_dtype()), learnable=True)


def build_sample(
    weight_defs: list[ValuesType], fact_def: ValuesType, n_weighted_neurons: int, n_features: tuple[int, int]
):
    n_weights = len(weight_defs)
    n_facts = n_weighted_neurons * n_weights

    a, b = n_features

    weights = [build_weight(100 + i, wd, a, b) for i, wd in enumerate(weight_defs)]

    factory = NeuronTestFactory(layers=LAYERS, id_provider_starts=0)

    return sources.from_dict(
        layers=LAYERS,
        neurons={
            16: [factory.create(16, value=build_value(fact_def, a)) for _ in range(n_facts)],
            13: [
                factory.create(13, inputs=[i * n_weights + j for j in range(n_weights)], weights=weights)
                for i in range(n_weighted_neurons)
            ],
        },
    )


@dataclass
class Flags:
    group_learnable_weight_parameters: bool
    optimize_linear_gathers: bool
    use_unique_pre_gathers: bool


FLAGS_PARAMETERS = [Flags(a, b, c) for a, b, c in itertools.product([False, True], [False, True], [False, True])]


def build_linear(sample: Network, period: int | None, flags: Flags):
    layer_sizes = {l.id: len(ns) for l, ns in sample.items()}
    return build_optimal_linear(network=sample, neurons=sample[13], layer_sizes=layer_sizes, period=period, **asdict(flags))


@dataclass
class Parameters:
    value_def: ValuesType
    weight_defs: list[ValuesType]
    period: int | None
    n_weighted_neurons: int
    n_input_features: int
    n_output_features: int


def build_all(p: Parameters, flags: Flags):
    n_features = p.n_input_features, p.n_output_features
    sample = build_sample(p.weight_defs, p.value_def, p.n_weighted_neurons, n_features)
    linear = build_linear(sample, p.period, flags)
    inputs = {"16": atleast_3d_rev(torch.stack(list(sample[16].get_values_torch()), dim=0))}

    if p.period is None:
        shape_expected = (p.n_weighted_neurons, p.n_output_features, 1)
    else:
        shape_expected = (p.n_weighted_neurons, p.period, p.n_output_features, 1)

    return linear, inputs, shape_expected


PARAMETERS = [
    Parameters("random", ["random", "random", "random"], 3, 5, 4, 3),
    Parameters("random", ["random", "random"], 2, 5, 4, 3),
    Parameters("random", ["random"], 1, 5, 4, 3),
    Parameters("random", ["random"], None, 5, 4, 3),
    Parameters("random", [1.0, 1.0, 1.0], 3, 5, 4, 3),
    Parameters("random", [1.0, 1.0], 2, 5, 4, 3),
    Parameters("random", [1.0], 1, 5, 4, 3),
    Parameters("random", [1.0], None, 5, 4, 3),
    Parameters("random", ["random", 1.0, 1.0], 3, 5, 4, 3),
    Parameters("random", ["random", 1.0], 2, 5, 4, 3),
    Parameters("random", ["random"], 1, 5, 4, 3),
    Parameters("random", ["random"], None, 5, 4, 3),
    Parameters(1.0, ["random", "random", "random"], 3, 5, 4, 3),
    Parameters(1.0, ["random", "random"], 2, 5, 4, 3),
    Parameters(1.0, ["random"], 1, 5, 4, 3),
    Parameters(1.0, ["random"], None, 5, 4, 3),
    Parameters(1.0, [1.0, 1.0, 1.0], 3, 5, 4, 3),
    Parameters(1.0, [1.0, 1.0], 2, 5, 4, 3),
    Parameters(1.0, [1.0], 1, 5, 4, 3),
    Parameters(1.0, [1.0], None, 5, 4, 3),
    Parameters(1.0, ["random", 1.0, 1.0], 3, 5, 4, 3),
    Parameters(1.0, ["random", 1.0], 2, 5, 4, 3),
    Parameters(1.0, ["random"], 1, 5, 4, 3),
    Parameters(1.0, ["random"], None, 5, 4, 3),
]


@pytest.mark.parametrize(["p", "flags"], itertools.product(PARAMETERS, FLAGS_PARAMETERS))
def test_shape(p: Parameters, flags: Flags):
    linear, inputs, shape_expected = build_all(p, flags)

    y = linear(inputs)
    print(linear)
    print("period", p.period)
    print("Expected shape:", shape_expected)
    print("Actual shape:", tuple(y.shape))
    assert y.shape == shape_expected


if __name__ == "__main__":
    test_shape(PARAMETERS[5], FLAGS_PARAMETERS[-1])
