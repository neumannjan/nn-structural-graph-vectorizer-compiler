from lib.utils import head_and_rest
from lib.vectorize.model import *
from lib.vectorize.pipeline.utils.chain_graph import ComputeChainGraph


class JoinSimpleLayerChains:
    def __init__(self, network: VectorizedOpSeqNetwork) -> None:
        self.network = network
        self._compute_chain_graph = ComputeChainGraph(network)

    def _for_layers(self, layers: dict[str, OperationSeq]):
        g = self._compute_chain_graph(layers)

        for chain in g.iter_chains(g):
            try:
                head, rest = head_and_rest(chain)
            except StopIteration:
                continue

            vals = layers[head]

            prev_r = head
            for r in rest:
                vals.operations.extend(layers[r].operations)
                layers[r] = vals
                del layers[prev_r]
                prev_r = r

    def join_simple_layer_chains(self):
        for batch in self.network.batches.values():
            self._for_layers(batch.layers)


def join_simple_layer_chains(network: VectorizedOpSeqNetwork):
    JoinSimpleLayerChains(network).join_simple_layer_chains()
    return network
