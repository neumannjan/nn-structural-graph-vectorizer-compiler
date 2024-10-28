from compute_graph_vectorize.sources.builders import from_neuralogic
from compute_graph_vectorize.sources.neuralogic_settings import NeuralogicSettings
from compute_graph_vectorize.vectorize.model.repr_aspython import prepr
from compute_graph_vectorize.vectorize.pipeline.build_initial_network import build_initial_network
from neuralogic.core import R, Transformation, V
from neuralogic.dataset import Dataset
from neuralogic.nn.base import Template


def dataset():
    dataset = Dataset()
    dataset.add_example(
        [
            R.edge(1, 2),
            R.edge(2, 1),
            R.a(1)[[1, 0, 0]],
            R.b(2)[[0, 1, 0]],
        ]
    )
    dataset.add_example(
        [
            R.edge(3, 4),
            R.a(3)[[1, 0, 0]],
            R.b(4)[[0, 1, 0]],
        ]
    )

    dataset.add_queries([R.predict(1)[0], R.predict(3)[1]])

    return dataset


def template():
    template = Template()

    template += R.l1(V.X)[2, 10] <= (R.a(V.X)[10, 3], R.b(V.Y)[10, 3])
    template += R.l1(V.X)[1, 5] <= R.a(V.X)[5, 3]
    template += (R.predict(V.X) <= R.l1(V.X)) | [Transformation.RELU]
    return template


if __name__ == "__main__":
    nsettings = NeuralogicSettings(iso_value_compression=False, chain_pruning=False)

    built_dataset = template().build(nsettings).build_dataset(dataset())

    network = from_neuralogic(built_dataset.samples, nsettings)
    vn = build_initial_network(network)
    print(prepr(vn))
