from lib.sources.base import Network
from lib.vectorize.model.network import VectorizedNetwork
from lib.vectorize.pipeline.build_initial_network import build_initial_network
from lib.vectorize.pipeline.compute_layer_counts import compute_layer_counts
from lib.vectorize.pipeline.compute_layer_shapes import compute_layer_shapes
from lib.vectorize.pipeline.concat_inputs_layers import concat_inputs_layers
from lib.vectorize.pipeline.merge_unit_facts import merge_unit_facts
from lib.vectorize.pipeline.reshape_fixed_count_reduce import reshape_fixed_count_reduce
from lib.vectorize.pipeline.utils.pipe import Pipe


def _print(a):
    print(a)
    return a


def build_vectorized_network(network: Network) -> VectorizedNetwork:
    return (
        Pipe(network)  #
        + build_initial_network
        + _print
        + merge_unit_facts
        + _print
        # + drop_unused_neurons  # TODO
        + compute_layer_counts
        # + transpose_fixed_count_linears  # <- optional
        + concat_inputs_layers  # <- gathers are expected starting here
        + reshape_fixed_count_reduce  # TODO: is this really necessary? maybe view shouldn't be a gather
        # + convert_linears_to_unique
        # + optimize_tail_gathers
        # + simplify_gathers
        + compute_layer_shapes  # <- shapes are expected starting here
        # + merge_weights
        # + materialize_unit_facts
    ).value
