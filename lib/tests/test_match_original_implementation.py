import jpype
import torch
from lib.datasets import MyMutagenesis
from lib.nn.topological.layers import (
    compute_neuron_ordinals,
    discover_all_layers,
    get_neurons_per_layer,
)
from lib.nn.topological.network_module import NetworkModule
from lib.nn.topological.settings import Settings
from lib.utils import value_to_tensor


def test_mutagenesis():
    try:
        dataset = MyMutagenesis()
        dataset.settings.compute_neuron_layer_indices = True
        # dataset.settings.iso_value_compression = False
        # dataset.settings.chain_pruning = False
        print("Building dataset...")
        _, built_dataset = dataset.build(sample_run=True)

        ###### CONFIG ######

        settings = Settings(
            # TODO assumptions
            check_same_layers_assumption=False,
        )

        torch.set_default_dtype(torch.float64)

        ###### DATASET CONFIG ######

        samples = built_dataset.samples

        ###### ALGORITHM ######

        print("Layers discovery...")
        layers = discover_all_layers(samples, settings)

        network = get_neurons_per_layer(samples)

        _, ordinals = compute_neuron_ordinals(layers, network, settings)

        model = NetworkModule(
            layers,
            network,
            ordinals,
            settings,
        )

        print(model)

        results: dict[int, torch.Tensor] = model()

        for layer in layers:
            expected = torch.squeeze(
                torch.stack([value_to_tensor(n.getRawState().getValue()) for n in network[layer.index]])
            )
            actual = torch.squeeze(results[layer.index])
            assert (torch.abs(expected - actual) < 1e-5).all(), (
                f"Values do not match at layer {layer.index} ({layer.type}). "
                f"Max difference is {torch.max(torch.abs(expected - actual))}. "
                f"Expected: {expected}\n"
                f"Actual: {actual}"
            )

        print("Expected:", expected)
        print("Actual:", actual)
        print("All values match!")

    except jpype.JException as e:
        print(e.message())
        print(e.stacktrace())
        raise e
