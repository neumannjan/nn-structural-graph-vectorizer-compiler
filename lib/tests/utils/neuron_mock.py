import itertools
from abc import abstractmethod
from typing import Iterator, Protocol, Sequence

from lib.nn.sources.minimal_api.dict import Neuron
from lib.nn.sources.base import LayerDefinition, LayerType, get_layer_id


class _BaseTestNeuronFactory:
    def __init__(self, id_provider: Iterator[int]) -> None:
        self._id_provider = id_provider

    @abstractmethod
    def _build_default(self) -> Neuron:
        pass

    def create(self, **kwargs):
        default = self._build_default()
        return Neuron.create_from(other=default, id=next(self._id_provider), **kwargs)


class _BaseTestNeuronFactoryProvider(Protocol):
    def __call__(self, id_provider: Iterator[int]) -> _BaseTestNeuronFactory:
        ...


class _FactTestNeuronFactory(_BaseTestNeuronFactory):
    def _build_default(self) -> Neuron:
        return Neuron(id=0, transformation="identity")


class _WeightedTestNeuronFactory(_BaseTestNeuronFactory):
    def _build_default(self) -> Neuron:
        return Neuron(id=0, transformation="tanh")


class _AggregationTestNeuronFactory(_BaseTestNeuronFactory):
    def _build_default(self) -> Neuron:
        return Neuron(id=0, transformation="identity")


_LAYER_FACTORIES: dict[LayerType, _BaseTestNeuronFactoryProvider] = {
    "FactLayer": _FactTestNeuronFactory,
    "AggregationLayer": _AggregationTestNeuronFactory,
    "WeightedAtomLayer": _WeightedTestNeuronFactory,
    "WeightedRuleLayer": _WeightedTestNeuronFactory,
}


class NeuronTestFactory:
    def __init__(self, layers: Sequence[LayerDefinition], id_provider_starts: Sequence[int] | int = 1000) -> None:
        if isinstance(id_provider_starts, Sequence):
            assert len(id_provider_starts) == len(layers)
            id_providers = [iter(itertools.count(s)) for s in id_provider_starts]
        elif isinstance(id_provider_starts, int):
            id_provider = iter(itertools.count(id_provider_starts))
            id_providers = [id_provider] * len(layers)
        else:
            raise ValueError()

        self._factory_per_layer: dict[int, _BaseTestNeuronFactory] = {
            layer.id: _LAYER_FACTORIES[layer.type](id_provider) for id_provider, layer in zip(id_providers, layers)
        }

    def for_layer(self, layer: LayerDefinition | int) -> _BaseTestNeuronFactory:
        layer_id = get_layer_id(layer)
        factory = self._factory_per_layer[layer_id]
        return factory

    def create(self, layer: LayerDefinition | int, **kwargs) -> Neuron:
        layer_id = get_layer_id(layer)
        factory = self._factory_per_layer[layer_id]
        return factory.create(**kwargs)
