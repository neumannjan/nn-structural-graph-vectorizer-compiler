from lib.sources.base import LayerDefinition, LayerOrdinal, Network
from lib.sources.views.map_ordinals import MapOrdinalsView
from lib.tests.utils.network_mock import generate_example_network


def build_view(network: Network, layer: LayerDefinition):
    view = MapOrdinalsView(network, {LayerOrdinal(layer.id, 0): LayerOrdinal(layer.id, 1)})
    return view


def test_map_ordinals_view_layer_ordinals():
    network = generate_example_network()

    l1 = network.layers.as_list()[1]

    view = build_view(network, l1)
    n_values = len(network[l1])

    print("Expected:", len(network[l1]))
    print("Actual:", len(view[l1]))
    assert len(network[l1]) == len(view[l1])

    expected = [LayerOrdinal(l1.id, 1 if i == 0 else i) for i in range(n_values)]
    actual = list(view[l1].ordinals)

    print("Expected:", len(expected), expected)
    print("Actual:", len(actual), actual)
    assert len(expected) == len(actual)
    assert all((a == e for a, e in zip(actual, expected)))


if __name__ == "__main__":
    test_map_ordinals_view_layer_ordinals()
