from typing import Mapping, Sequence

from neuralogic.core.builder.builder import NeuralSample

from compute_graph_vectorize.sources.base import LayerDefinition
from compute_graph_vectorize.sources.minimal_api.base import MinimalAPINetwork
from compute_graph_vectorize.sources.minimal_api.dict import MinimalAPIDictNetwork, Neuron
from compute_graph_vectorize.sources.minimal_api.internal.neuralogic import JavaNeuron
from compute_graph_vectorize.sources.minimal_api.neuralogic import MinimalAPINeuralogicNetwork
from compute_graph_vectorize.sources.minimal_api.ordinals import MinimalAPIOrdinals
from compute_graph_vectorize.sources.minimal_api_bridge import NetworkImpl
from compute_graph_vectorize.sources.neuralogic_settings import NeuralogicSettings


def from_minimal_api(minimal_api: MinimalAPINetwork, custom_ordinals: MinimalAPIOrdinals | None = None):
    """
    Get a Network from an arbitrary "minimal API" representation of a network.

    For more details see the documentation for `MinimalAPINetwork` and `MinimalAPIOrdinals`.

    If `custom_ordinals` is not populated, the default `MinimalAPIOrdinalsImpl` is created.
    """
    return NetworkImpl(minimal_api=minimal_api, custom_ordinals=custom_ordinals)


def from_neuralogic(samples: Sequence[NeuralSample | JavaNeuron], settings: NeuralogicSettings):
    """
    Get a Network from a set of NeuraLogic neural (built) samples.

    Uses the underlying NeuraLogic Java representations under the hood.
    """
    minimal = MinimalAPINeuralogicNetwork(samples=samples, settings=settings)
    return from_minimal_api(minimal_api=minimal, custom_ordinals=None)


def from_dict(layers: Sequence[LayerDefinition], neurons: Sequence[Sequence[Neuron]] | Mapping[str, Sequence[Neuron]]):
    """Get a Network from a pure Python data representation."""
    minimal = MinimalAPIDictNetwork(layers=layers, neurons=neurons)
    return from_minimal_api(minimal_api=minimal, custom_ordinals=None)
