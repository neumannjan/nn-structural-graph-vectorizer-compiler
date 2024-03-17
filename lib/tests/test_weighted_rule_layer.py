from typing import Sequence

import numpy as np
import pytest
import torch
from lib.nn import sources
from lib.nn.sources.base import LayerDefinition
from lib.nn.sources.minimal_api.dict import WeightDefinitionImpl
from lib.nn.topological.layer import Layer
from lib.nn.topological.settings import Settings
from lib.tests.utils.neuron_factory import NeuronTestFactory
from lib.tests.utils.test_params import SETTINGS_PARAMS

LAYERS = [
    LayerDefinition(16, "FactLayer"),
    LayerDefinition(13, "WeightedRuleLayer"),
    LayerDefinition(12, "AggregationLayer"),
]

UNIT_WEIGHT = WeightDefinitionImpl(0, torch.tensor([1.0]), learnable=False)
WEIGHTS = {
    15: WeightDefinitionImpl(
        15,
        torch.tensor(
            [  #
                [0.95, 0.2, 0.54],
                [-0.81, -0.09, 0.23],
                [-0.67, 0.67, 0.24],
            ]
        ),
        learnable=True,
    ),
    14: WeightDefinitionImpl(
        14,
        torch.tensor(
            [  #
                [0.24, -0.96, -0.94],
                [-0.25, -0.17, -0.94],
                [0.56, -0.73, 0.27],
            ]
        ),
        learnable=True,
    ),
}


def build_sample(weights: list[WeightDefinitionImpl], facts_values: Sequence[np.ndarray]):
    assert len(facts_values) % len(weights) == 0
    n_weights = len(weights)
    n_neurons = len(facts_values) // len(weights)

    factory = NeuronTestFactory(layers=LAYERS, id_provider_starts=0)

    return sources.from_dict(
        layers=LAYERS,
        neurons={
            16: [factory.create(16, value=v) for v in facts_values],
            13: [
                factory.create(13, inputs=[i * n_weights + j for j in range(n_weights)], weights=weights)
                for i in range(n_neurons)
            ],
            12: [factory.create(12, inputs=[n_weights * n_neurons + i for i in range(n_neurons)])],
        },
    )


FACTS_VALUES = np.array(
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
)


@pytest.mark.parametrize("settings", SETTINGS_PARAMS)
def test_weighted_rule_layer(settings: Settings):
    network = build_sample(
        weights=[UNIT_WEIGHT, WEIGHTS[14], WEIGHTS[15], UNIT_WEIGHT], facts_values=list(FACTS_VALUES)
    )
    inputs = {"16": torch.tensor(FACTS_VALUES, dtype=torch.get_default_dtype()).unsqueeze(-1)}
    layer = Layer.from_network(out_to=13, network=network, neurons=network[13], settings=settings)

    print(layer)

    expected = torch.tensor(
        [[0.85, -0.53, 0.72], [0.85, -0.53, 0.72], [0.85, -0.53, 0.72], [0.85, -0.53, 0.72], [0.85, -0.53, 0.72]]
    ).unsqueeze(-1)
    actual = layer(inputs)["13"]

    print("expected", expected)
    print("actual", actual)
    print("expected shape", expected.shape)
    print("actual shape", actual.shape)

    assert ((expected - actual).abs() <= 0.01).all()


if __name__ == "__main__":
    test_weighted_rule_layer(SETTINGS_PARAMS[0])
