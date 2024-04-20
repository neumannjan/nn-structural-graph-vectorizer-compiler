from lib.utils import head_and_rest
from lib.vectorize.model import *
from lib.vectorize.pipeline.utils.chain_graph import ComputeChainGraph


class JoinSimpleLayerChains:
    def __init__(self, network: VectorizedOpSeqNetwork) -> None:
        self.network = network
        self._compute_chain_graph = ComputeChainGraph(network, types=(LayerRefs.TYPE_LAYER,))

    def _for_layers(self, layers: dict[str, OperationSeq]):
        g = self._compute_chain_graph(layers)

        for chain in g.iter_chains():
            try:
                (_, head), rest = head_and_rest(chain)
            except StopIteration:
                continue

            vals = layers[head]

            prev_r = head
            for (_, r) in rest:
                vals.expected_count = layers[r].expected_count
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
