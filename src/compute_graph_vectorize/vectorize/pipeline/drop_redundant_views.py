from collections.abc import Iterable

from compute_graph_vectorize.vectorize.model import *


class DropRedundantViews:
    def __init__(self, network: VectorizedOpSeqNetwork) -> None:
        self.network = network

    def _get_filtered_ops(self, ops: OperationSeq) -> Iterable[Operation]:
        view: View | None = None
        it = iter(ops)

        try:
            while True:
                op = next(it)

                if isinstance(op, View):
                    view = op
                else:
                    if view is not None:
                        yield view
                        view = None

                    yield op
        except StopIteration:
            pass

        if view is not None:
            yield view

    def drop_redundant_views(self):
        for batch_id, batch in self.network.batches.items():
            for layer_id, ops in batch.layers.items():
                ops.operations = list(self._get_filtered_ops(ops))


def drop_redundant_views(network: VectorizedOpSeqNetwork):
    DropRedundantViews(network).drop_redundant_views()
    return network
