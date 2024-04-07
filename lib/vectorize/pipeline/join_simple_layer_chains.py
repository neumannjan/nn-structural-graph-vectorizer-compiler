from collections import defaultdict
from typing import Collection, Iterable, Literal, Mapping

from lib.utils import head_and_rest
from lib.vectorize.model import *


class _ChainGraph:
    __slots__ = ("nodes", "single_successors", "single_predecessors")

    def __init__(
        self,
        nodes: Collection[str],
        single_successors: Mapping[str, str],
        single_predecessors: Mapping[str, str],
    ) -> None:
        self.nodes = nodes
        self.single_successors = single_successors
        self.single_predecessors = single_predecessors


class JoinSimpleLayerChains:
    def __init__(self, network: VectorizedOpSeqNetwork) -> None:
        self.network = network

    def _iter_layer_refs(self, ops: OperationSeq) -> Iterable[str]:
        if ops.layer_refs is not None:
            yield from ops.layer_refs.layers

        for op in ops.operations:
            match op:
                case Linear(weight_ops=OperationSeq(layer_refs=LayerRefs(layers=layers))):
                    yield from layers

    def _iter_all_layer_refs(self, layers: dict[str, OperationSeq]) -> Iterable[str]:
        for ops in layers.values():
            yield from self._iter_layer_refs(ops)

    def _compute_chain_graph(self, layers: dict[str, OperationSeq]) -> _ChainGraph:
        v = list(layers.keys())
        succ: dict[str, str] = {}

        for layer_id, ops in layers.items():
            match ops:
                case OperationSeq(layer_refs=LayerRefs(facts=[], weights=[], layers=[ref_layer_id])):
                    succ[ref_layer_id] = layer_id

        visited: set[str] = set()

        for ref in self._iter_all_layer_refs(layers):
            if ref not in visited:
                visited.add(ref)
            else:
                succ.pop(ref, None)

        pred: dict[str, str] = {b: a for a, b in succ.items()}

        return _ChainGraph(v, succ, pred)

    def _iter_beginnings(self, g: _ChainGraph) -> Iterable[str]:
        for n in g.nodes:
            if n not in g.single_predecessors:
                yield n

    def _iter_single_chain(self, g: _ChainGraph, beginning: str) -> Iterable[str]:
        n = beginning
        yield n

        while True:
            try:
                n = g.single_successors[n]
            except KeyError:
                return
            yield n

    def _iter_chains(self, g: _ChainGraph) -> Iterable[Iterable[str]]:
        for n in self._iter_beginnings(g):
            yield self._iter_single_chain(g, beginning=n)

    def _iter_layer_pairs(self, g: _ChainGraph) -> Iterable[tuple[str, str]]:
        for chain in self._iter_chains(g):
            try:
                head, rest = head_and_rest(chain)
            except StopIteration:
                continue

            for r in rest:
                yield head, r

    def _for_layers(self, layers: dict[str, OperationSeq]):
        g = self._compute_chain_graph(layers)

        for chain in self._iter_chains(g):
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
