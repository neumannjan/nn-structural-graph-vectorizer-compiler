import itertools

import numpy as np
import pytest
import torch
from lib.nn.topological.layers import LayerDefinition, compute_neuron_ordinals, get_neurons_per_layer
from lib.nn.topological.settings import Settings
from lib.nn.topological.weighted_rule_layer import WeightedRuleLayer
from lib.tests.utils.network_mock import MockJavaNeuron, MockJavaWeight
from lib.tests.utils.test_params import SETTINGS_PARAMS

LAYERS = [
    LayerDefinition("", 16),
    LayerDefinition("", 13),
    LayerDefinition("", 12),
]

UNIT_WEIGHT = MockJavaWeight(0, np.array([1]), learnable=False)
WEIGHTS = {
    15: MockJavaWeight(
        15,
        np.array(
            [  #
                [0.95, 0.2, 0.54],
                [-0.81, -0.09, 0.23],
                [-0.67, 0.67, 0.24],
            ]
        ),
        learnable=True,
    ),
    14: MockJavaWeight(
        14,
        np.array(
            [  #
                [0.24, -0.96, -0.94],
                [-0.25, -0.17, -0.94],
                [0.56, -0.73, 0.27],
            ]
        ),
        learnable=True,
    ),
}


def build_sample(weights: list[MockJavaWeight], n_neurons: int):
    n_weights = len(weights)

    index_factory = iter(itertools.count())

    def _n():
        return next(index_factory)

    return MockJavaNeuron(
        _n(),
        16,
        [
            MockJavaNeuron(
                _n(),
                13,
                [  #
                    MockJavaNeuron(_n(), 12) for _ in range(n_weights)
                ],
                weights,
            )
            for _ in range(n_neurons)
        ],
    )


@pytest.mark.parametrize(["settings"], [[settings] for settings in SETTINGS_PARAMS])
def test_weighted_rule_layer(settings: Settings):
    sample = build_sample(weights=[UNIT_WEIGHT, WEIGHTS[14], WEIGHTS[15], UNIT_WEIGHT], n_neurons=5)
    network = get_neurons_per_layer([sample])
    inputs = {
        12: torch.tensor(
            [
                [-0.12, -0.62, -0.38],
                [0.65, 0.11, 0.2],
                [0.59, 0.62, -0.31],
                [1.0, 1.0, 1.0],  #
                [-0.12, -0.62, -0.38],
                [0.65, 0.11, 0.2],
                [0.59, 0.62, -0.31],
                [1.0, 1.0, 1.0],  #
                [-0.12, -0.62, -0.38],
                [0.65, 0.11, 0.2],
                [0.59, 0.62, -0.31],
                [1.0, 1.0, 1.0],  #
                [-0.12, -0.62, -0.38],
                [0.65, 0.11, 0.2],
                [0.59, 0.62, -0.31],
                [1.0, 1.0, 1.0],  #
                [-0.12, -0.62, -0.38],
                [0.65, 0.11, 0.2],
                [0.59, 0.62, -0.31],
                [1.0, 1.0, 1.0],  #
            ]
        ).unsqueeze(-1)
    }
    _, ordinals = compute_neuron_ordinals(LAYERS, network, settings)
    print(ordinals)
    layer = WeightedRuleLayer(
        layer_neurons=sample.getInputs(),
        neuron_ordinals=ordinals,
        assume_rule_weights_same=settings.assume_rule_weights_same,
        check_same_inputs_dim_assumption=settings.check_same_inputs_dim_assumption,
    )

    print(layer)

    expected = torch.tensor(
        [[0.85, -0.53, 0.72], [0.85, -0.53, 0.72], [0.85, -0.53, 0.72], [0.85, -0.53, 0.72], [0.85, -0.53, 0.72]]
    ).unsqueeze(-1)
    actual = layer(inputs)

    print("expected", expected)
    print("actual", actual)
    print("expected shape", expected.shape)
    print("actual shape", actual.shape)

    assert ((expected - actual).abs() <= 0.01).all()
