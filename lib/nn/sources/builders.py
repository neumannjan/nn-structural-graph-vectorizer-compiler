from typing import Mapping, Sequence

from neuralogic.core.builder.builder import NeuralSample

from lib.nn.sources.base import LayerDefinition
from lib.nn.sources.minimal_api.base import MinimalAPINetwork
from lib.nn.sources.minimal_api.dict import MinimalAPIDictNetwork, Neuron
from lib.nn.sources.minimal_api.internal.java import JavaNeuron
from lib.nn.sources.minimal_api.java import MinimalAPIJavaNetwork
from lib.nn.sources.minimal_api.ordinals import MinimalAPIOrdinals
from lib.nn.sources.minimal_api_bridge import NetworkImpl
from lib.nn.topological.settings import Settings


def from_minimal_api(minimal_api: MinimalAPINetwork, custom_ordinals: MinimalAPIOrdinals | None = None):
    """
    Get a Network from an arbitrary "minimal API" representation of a network.

    For more details see the documentation for `MinimalAPINetwork` and `MinimalAPIOrdinals`.

    If `custom_ordinals` is not populated, the default `MinimalAPIOrdinalsImpl` is created.
    """
    return NetworkImpl(minimal_api=minimal_api, custom_ordinals=custom_ordinals)


def from_java(samples: Sequence[NeuralSample | JavaNeuron], settings: Settings):
    """
    Get a Network from a set of NeuraLogic neural (built) samples.

    Uses the underlying NeuraLogic Java representations under the hood.
    """
    minimal = MinimalAPIJavaNetwork(samples=samples, settings=settings)
    return from_minimal_api(minimal_api=minimal, custom_ordinals=None)


def from_dict(layers: Sequence[LayerDefinition], neurons: Sequence[Sequence[Neuron]] | Mapping[int, Sequence[Neuron]]):
    """Get a Network from a pure Python data representation."""
    minimal = MinimalAPIDictNetwork(layers=layers, neurons=neurons)
    return from_minimal_api(minimal_api=minimal, custom_ordinals=None)
