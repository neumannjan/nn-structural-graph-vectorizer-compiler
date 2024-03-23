from lib.vectorize.model import *


class DropUnusedNeurons:
    def __init__(self, network: VectorizedNetwork) -> None:
        self.network = network

    def _find_used_neurons(self):
        facts: set[Ref] = set()
        batch_neurons: dict[int, set[Ref]] = {}

        for bid, batch in self.network.batches.items():
            batch_neurons[bid] = ns = set()
            for layer in batch.layers.values():
                match layer.base:
                    case InputLayerBase(input=Refs(refs)):
                        facts.update((r for r in refs if isinstance(r, FactRef)))
                        ns.update((r for r in refs if isinstance(r, LayerRef)))
                    case LinearLayerBase(input=Refs(refs), weight=Refs(wrefs)):
                        facts.update((r for r in refs if isinstance(r, FactRef)))
                        facts.update((r for r in wrefs if isinstance(r, FactRef)))
                        ns.update((r for r in refs if isinstance(r, LayerRef)))
                        ns.update((r for r in wrefs if isinstance(r, LayerRef)))
                    case LinearGatherLayerBase(input=Refs(refs), weight=Refs(wrefs)):
                        facts.update((r for r in refs if isinstance(r, FactRef)))
                        facts.update((r for r in wrefs if isinstance(r, FactRef)))
                        ns.update((r for r in refs if isinstance(r, LayerRef)))
                        ns.update((r for r in wrefs if isinstance(r, LayerRef)))
                    case _:
                        assert False

        self._used_facts = facts
        self._batch_neurons = batch_neurons

    def drop_unused_neurons(self):
        self._find_used_neurons()

        ref_map: dict[Ref, Ref] = {}

        for lid, layer in self.network.fact_layers.items():
            for o, fact in enumerate(layer.facts):
                if fact in self._used_facts:
                    pass

        # TODO

        for bid, batch in self.network.batches.items():
            ns = self._batch_neurons[bid]
            for lid, layer in batch.layers.items():
                pass


def drop_unused_neurons(network: VectorizedNetwork):
    DropUnusedNeurons(network).drop_unused_neurons()
    return network
