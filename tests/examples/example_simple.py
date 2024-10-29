from compute_graph_vectorize.sources.builders import from_neuralogic
from compute_graph_vectorize.sources.neuralogic_settings import NeuralogicSettings
from compute_graph_vectorize.vectorize.model.repr_aspython import prepr
from compute_graph_vectorize.vectorize.pipeline.build_initial_network import build_initial_network
from compute_graph_vectorize.vectorize.pipeline.compute_layer_counts import compute_layer_counts
from compute_graph_vectorize.vectorize.pipeline.compute_layer_shapes import compute_layer_shapes
from compute_graph_vectorize.vectorize.pipeline.dissolve_identity_layers import predissolve_identity_layers
from compute_graph_vectorize.vectorize.pipeline.merge_same_value_facts import merge_same_value_facts
from compute_graph_vectorize.vectorize.pipeline.merge_unit_facts import merge_unit_facts
from compute_graph_vectorize.vectorize.pipeline.utils.pipe import PIPE
from neuralogic.core import R, V
from neuralogic.dataset import Dataset
from neuralogic.nn.base import Template


def dataset():
    dataset = Dataset()
    dataset.add_example(
        [
            R.edge(1, 2),
            R.edge(2, 1),
            R.edge(1, 3),
            R.edge(2, 3),
            R.edge(3, 2),
            R.a(1),
            R.a(2),
            R.a(3),
        ]
    )
    dataset.add_example(
        [
            R.edge(3, 4),
            R.a(3),
            R.a(4),
        ]
    )

    dataset.add_queries([R.predict(1)[0], R.predict(3)[1]])

    return dataset


def template():
    template = Template()

    template += R.l1_a(V.X)[10, 10] <= (R.a(V.X), R.a(V.Y), R.edge(V.X, V.Y))
    template += R.l1_b(V.X)[5, 10] <= R.a(V.X)
    template += R.predict(V.X)[1, 10] <= (R.l1_a(V.X)[10, 10], R.l1_b(V.X)[10, 5])
    return template


if __name__ == "__main__":
    nsettings = NeuralogicSettings(iso_value_compression=False, chain_pruning=False)

    built_dataset = template().build(nsettings).build_dataset(dataset())

    network = from_neuralogic(built_dataset.samples, nsettings)
    func = (PIPE
        + build_initial_network
        + merge_unit_facts
        + merge_same_value_facts
        + predissolve_identity_layers
        # + compute_layer_shapes
        # + compute_layer_counts
        # + build_separate_input_refs(ShapeLayerIndexer)
    )

    vn = func(network)
    print(prepr(vn))
