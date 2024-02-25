from typing import Sequence

from neuralogic.core.builder.builder import NeuralSample

from lib.nn.sources.internal.java_source import JavaNeuron
from lib.nn.sources.java_source import JavaNeuralNetworkDefinition
from lib.nn.topological.settings import Settings


def from_java(samples: Sequence[NeuralSample | JavaNeuron], settings: Settings):
    return JavaNeuralNetworkDefinition(samples, settings)
