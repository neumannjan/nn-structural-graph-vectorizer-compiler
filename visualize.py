from collections import deque
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence, TypedDict

import jpype
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from lib.datasets import MyMutagenesis
from lib.nn.definitions.settings import Settings
from neuralogic.core.builder.builder import NeuralSample


class Neuron(TypedDict):
    name: str
    inner_type: str
    layer: int
    weight_indices: list[int]
    offset_index: int
    weight_values: list[str]
    offset_value: str


WEIGHTED_NEURON_CLASS = None


def get_neuron(java_neuron) -> tuple[int, Neuron]:
    global WEIGHTED_NEURON_CLASS
    if WEIGHTED_NEURON_CLASS is None:
        WEIGHTED_NEURON_CLASS = jpype.JClass(
            "cz.cvut.fel.ida.neural.networks.structure.components.neurons.WeightedNeuron"
        )

    if isinstance(java_neuron, WEIGHTED_NEURON_CLASS):
        weight_indices = [w.index for w in java_neuron.getWeights()]
        offset_index = java_neuron.getOffset().index
        weight_values = [w.value.toString() for w in java_neuron.getWeights()]
        offset_value = java_neuron.getOffset().value.toString()
    else:
        weight_indices = []
        offset_index = -1
        weight_values = []
        offset_value = '0'

    return (
        java_neuron.getIndex(),
        Neuron(
            name=java_neuron.getName(),
            inner_type=java_neuron.getClass().getSimpleName(),
            layer=java_neuron.getLayer(),
            weight_indices=weight_indices,
            offset_index=offset_index,
            weight_values=weight_values,
            offset_value=offset_value,
        ),
    )


def iterate_neurons(java_neuron) -> Iterable[tuple[int, Neuron]]:
    queue = deque([java_neuron])
    while len(queue) > 0:
        n = queue.popleft()
        yield get_neuron(n)

        queue.extend(n.getInputs())


def iterate_edges(java_neuron) -> Iterable[tuple[int, int]]:
    queue = deque([java_neuron])
    while len(queue) > 0:
        n_right = queue.popleft()

        inputs = n_right.getInputs()

        for n_left in inputs:
            yield (n_left.getIndex(), n_right.getIndex())

        queue.extend(inputs)


def map_to_ints(values: list) -> tuple[list[int], dict[Any, int]]:
    val_set = set(values)
    val_to_idx_map = {v: i for i, v in enumerate(val_set)}
    return [val_to_idx_map[v] for v in values], val_to_idx_map


def get_as_colors(values: list, cmap, n: int | None = None) -> tuple[list, list, Sequence]:
    val_set = sorted(set(values))

    out_keys = list(val_set)

    if n is not None and len(val_set) < n:
        val_set += ((object(), None) for _ in range(n - len(val_set)))

    val_to_idx_map = {v: i for i, v in enumerate(val_set)}
    norm = plt.Normalize()
    colors = cmap(norm(list(val_to_idx_map.values())))

    out = [colors[val_to_idx_map[v]] for v in values]

    return out, out_keys, colors


class Graph:
    def __init__(self, neural_sample: NeuralSample, reindex=True) -> None:
        start_neuron = neural_sample.java_sample.query.neuron

        g = nx.DiGraph()
        g.add_nodes_from(iterate_neurons(start_neuron))
        g.add_edges_from(iterate_edges(start_neuron))

        self._fix_topological_generations(g)
        if reindex:
            g = self._reindex(g)

        self.g = g

    def _fix_topological_generations(self, g: nx.DiGraph):
        layers = set((d["layer"] for _, d in g.nodes(data=True)))
        layers_map: dict[int, int] = dict()

        for i, l_original in enumerate(sorted(layers, reverse=True)):
            layers_map[l_original] = i

        for _, d in g.nodes(data=True):
            d["layer"] = layers_map[d["layer"]]

    def _reindex(self, g: nx.DiGraph):
        label_mapping = {j: i for i, j in enumerate(nx.algorithms.dag.topological_sort(g))}

        for n, d in g.nodes(data=True):
            d["orig_index"] = n

        return nx.relabel.relabel_nodes(g, label_mapping)


def draw_graph(
    g: nx.DiGraph,
    layer_key="layer",
    color_key="inner_type",
    label_key: str | None = "layer",
    edge_color: Literal["source", "target"] | None = "target",
):
    pos = nx.drawing.layout.multipartite_layout(g, subset_key=layer_key)

    fig, ax = plt.subplots()

    color_keys = [d[color_key] for _, d in g.nodes(data=True)]
    node_cmap = plt.cm.tab10
    node_color_vals, types_uniq, node_colors = get_as_colors(color_keys, cmap=node_cmap, n=10)

    if edge_color == "source":
        edge_color_keys = [g.nodes[s]["layer"] for s, _ in g.edges]
    elif edge_color == "target":
        edge_color_keys = [g.nodes[t]["layer"] for _, t in g.edges]
    elif edge_color is not None:
        raise NotImplementedError()

    if edge_color is None:
        edge_color_vals = None
    else:
        edge_color_vals, _ = map_to_ints(edge_color_keys)

    nx.drawing.nx_pylab.draw_networkx(
        g,
        pos=pos,
        ax=ax,
        node_color=node_color_vals,
        edge_color=edge_color_vals,
        edge_cmap=plt.cm.tab10,
        with_labels=label_key is not None,
        labels={i: d[label_key] for i, d in g.nodes(data=True)} if label_key != "index" else None,
    )
    ax.set_title("DAG layout in topological order")
    fig.tight_layout()
    legend_elements = [
        mpl.lines.Line2D([0, 1], [0, 0], marker="o", color="w", label=t, markerfacecolor=c, markersize=15)
        for t, c in zip(types_uniq, node_colors)
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    return fig


class NeuronSetGraph:
    def __init__(self, graph: Graph) -> None:
        orig_g = graph.g
        g = nx.DiGraph()
        g.add_nodes_from(((i, d) for i, d in orig_g.nodes(data=True)))

        # self._double_topological_ordering(g)
        g.add_edges_from(orig_g.edges)
        self._compute_input_subsets(orig_g, g)

        self.g = g

    def _double_topological_ordering(self, g: nx.DiGraph):
        for _, d in g.nodes(data=True):
            d["layer"] *= 2

    def _compute_input_subsets(self, orig_g: nx.DiGraph, g: nx.DiGraph):
        for n, d in orig_g.nodes(data=True):
            subset_key = tuple(orig_g.predecessors(n))

            g.nodes[n]["shortname"] = f"{n}\n{str(tuple(subset_key))}"
            g.nodes[n]["weight_label"] = f"{d['weight_indices']} {d['offset_index']}"
            # g.nodes[n]["weight_label"] = f"{d['weight_values']} {d['offset_value']}"

            # if len(subset_key) == 0:
            #     continue
            # else:
            #     g.add_node(
            #         subset_key, inner_type="subset", shortname=str(tuple(subset_key)), layer=g.nodes[n]["layer"] - 1
            #     )
            #     g.add_edge(subset_key, n)

            # for n_p in subset_key:
            #     g.add_edge(n_p, subset_key)


def do_sample(neural_sample: NeuralSample, reindex=True, stage: Literal[0, 1] = 1):
    graph = Graph(neural_sample, reindex=reindex)

    if stage == 0:
        draw_graph(graph.g)
        return

    graph = NeuronSetGraph(graph)

    if stage == 1:
        draw_graph(graph.g, label_key="shortname", edge_color="source")
        # draw_graph(graph.g, label_key="weight_label", edge_color="source")
        return


if __name__ == "__main__":
    try:
        settings = Settings()
        dataset = MyMutagenesis(settings)
        # settings.neuralogic.iso_value_compression = False
        # settings.neuralogic.chain_pruning = False
        built_dataset = dataset.build(sample_run=True)

        out_dir = Path(f"./imgs/{dataset.name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        i = 108
        # built_dataset.samples[i].draw()
        do_sample(built_dataset.samples[i], reindex=True, stage=1)
        plt.show()

        # for i, sample in tqdm(enumerate(built_dataset.samples), total=len(built_dataset.samples)):
        #     do_sample(sample)
        #     plt.savefig(out_dir / f"{i}.jpg")

    except jpype.JException as e:
        print(e.message())
        print(e.stacktrace())
        raise e
