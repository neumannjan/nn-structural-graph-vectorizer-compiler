import random
from typing import Literal

import numpy as np
import pytest
from lib.nn import sources
from lib.nn.gather import SingleLayerGather, TakeValue, build_optimal_multi_layer_gather
from lib.nn.sources.base import LayerDefinition, LayerOrdinal
from lib.nn.sources.views.merge_facts import MergeFactsView
from lib.tests.utils.neuron_factory import NeuronTestFactory

VALUES = [np.array([1.0]), np.array([2.0]), np.array([3.0])]

TOTAL_NEURONS = 100


LAYERS: list[LayerDefinition] = [LayerDefinition(16, "FactLayer"), LayerDefinition(13, "AggregationLayer")]


def build_network(n_layers: Literal[1, 2], n_values: int):
    assert n_values <= len(VALUES)

    factory = NeuronTestFactory(layers=LAYERS[:n_layers])

    neurons_layer1_grouped_by_len_values = [
        [factory.create(LAYERS[0].id, value=v.copy()) for v in VALUES[:n_values]] for _ in range(TOTAL_NEURONS)
    ]

    neurons_layer1 = [n for ns in neurons_layer1_grouped_by_len_values for n in ns]

    if n_layers == 1:
        return sources.from_dict(layers=LAYERS[:n_layers], neurons=[neurons_layer1])

    neurons_layer2 = [
        factory.create(LAYERS[1].id, inputs=[n.id for n in random.choice(neurons_layer1_grouped_by_len_values)])
        for _ in range(len(neurons_layer1_grouped_by_len_values))
    ]

    if n_layers == 2:
        return sources.from_dict(layers=LAYERS[:n_layers], neurons=[neurons_layer1, neurons_layer2])


def test_merge_facts_view_layer_ordinals():
    n_values = len(VALUES)
    network = build_network(n_layers=1, n_values=n_values)

    view = MergeFactsView(network)

    expected = [LayerOrdinal(LAYERS[0].id, i) for i in range(n_values)]
    actual = list(view[16].ordinals)

    assert len(actual) == len(expected)
    assert all((a == e for a, e in zip(actual, expected)))


def test_merge_facts_view_all_ordinals():
    n_values = len(VALUES)
    network = build_network(n_layers=1, n_values=n_values)

    view = MergeFactsView(network)

    expected = [LayerOrdinal(LAYERS[0].id, i) for i in range(n_values)]
    actual = list(view.ordinals)

    assert len(actual) == len(expected)
    assert all((a == e for a, e in zip(actual, expected)))


def test_merge_facts_view_inputs_ordinals():
    n_values = len(VALUES)
    network = build_network(n_layers=2, n_values=n_values)

    view = MergeFactsView(network)

    expected = [LayerOrdinal(LAYERS[0].id, i % n_values) for i in range(TOTAL_NEURONS * n_values)]
    actual = list(view[LAYERS[1]].inputs.ordinals)
    assert len(actual) == len(expected)
    assert all((a == e for a, e in zip(actual, expected)))

    expected = [LayerOrdinal(LAYERS[0].id, i) for i in range(n_values)]
    actual = list(view[LAYERS[0]].ordinals)
    assert len(actual) == len(expected)
    assert all((a == e for a, e in zip(actual, expected)))


@pytest.mark.parametrize(["use_unique_pre_gathers"], [[False], [True]])
def test_gather_not_merged(use_unique_pre_gathers: bool):
    n_values = len(VALUES)
    network = build_network(n_layers=2, n_values=n_values)
    layer_sizes = {l.id: len(ns) for l, ns in network.items()}

    gather = build_optimal_multi_layer_gather(
        inputs_ordinals=list(network[LAYERS[1].id].inputs.ordinals),
        layer_sizes=layer_sizes,
        use_unique_pre_gathers=use_unique_pre_gathers,
    )

    assert gather.total_items == TOTAL_NEURONS * n_values
    assert gather.optimal_period == TOTAL_NEURONS * n_values


@pytest.mark.parametrize(["use_unique_pre_gathers"], [[False], [True]])
def test_gather_merged(use_unique_pre_gathers: bool):
    n_values = len(VALUES)
    network = build_network(n_layers=2, n_values=n_values)

    view = MergeFactsView(network)
    layer_sizes = {l.id: len(ns) for l, ns in view.items()}

    gather = build_optimal_multi_layer_gather(
        inputs_ordinals=list(view[LAYERS[1].id].inputs.ordinals),
        layer_sizes=layer_sizes,
        use_unique_pre_gathers=use_unique_pre_gathers,
    )

    assert gather.total_items == TOTAL_NEURONS * n_values
    assert gather.optimal_period == n_values
    assert gather.get_optimal().total_items == n_values


@pytest.mark.parametrize(["use_unique_pre_gathers"], [[False], [True]])
def test_gather_merged_single(use_unique_pre_gathers: bool):
    n_values = 1
    network = build_network(n_layers=2, n_values=n_values)

    view = MergeFactsView(network)
    layer_sizes = {l.id: len(ns) for l, ns in view.items()}

    gather = build_optimal_multi_layer_gather(
        inputs_ordinals=list(view[LAYERS[1].id].inputs.ordinals),
        layer_sizes=layer_sizes,
        use_unique_pre_gathers=use_unique_pre_gathers,
    )

    assert gather.total_items == 1
    assert gather.optimal_period == 1
    assert gather.get_optimal().total_items == 1
    assert isinstance(gather, SingleLayerGather) and isinstance(gather.delegate, TakeValue)


if __name__ == "__main__":
    test_merge_facts_view_inputs_ordinals()
