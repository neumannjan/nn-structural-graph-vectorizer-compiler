from expression import pipe

from lib.sources.base import Network
from lib.vectorize.model.network import VectorizedNetwork
from lib.vectorize.pipeline.build_initial_network import build_initial_network
from lib.vectorize.pipeline.compute_layer_shapes import compute_shapes


def _print(a):
    print(a)
    return a


def build_vectorized_network(network: Network) -> VectorizedNetwork:
    return pipe(
        network,
        build_initial_network,
        # merge_unit_facts,
        _print,
        # drop_unused_neurons,
        # transpose_fixed_count_linears,  # <- optional
        compute_shapes,
        # # <- shapes are expected starting here
        # combine_inputs_layers
        # # <- gathers are expected starting here
        # reshape_fixed_count_reduce,
        # convert_linears_to_unique,
        # optimize_tail_gathers,
        # simplify_gathers,
        # merge_weights,
        # materialize_unit_facts,
    )
