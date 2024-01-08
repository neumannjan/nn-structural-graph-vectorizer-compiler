from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import jpype
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from lib.datasets import MyMutagenesis
from neuralogic.core.builder.builder import NeuralSample

d = MyMutagenesis()


@dataclass(frozen=True)
class Neuron:
    index: int
    name: str
    type: str
    layer: int

    @classmethod
    def build(cls, java_neuron) -> "Neuron":
        return Neuron(
            index=java_neuron.getIndex(),
            name=java_neuron.getName(),
            type=java_neuron.getClass().getSimpleName(),
            layer=java_neuron.getLayer(),
        )

    def serialize(self) -> dict[str, Any]:
        data = asdict(self)
        del data["index"]
        return data

    def __repr__(self) -> str:
        return f"<{self.index}>"


@dataclass(frozen=True)
class Edge:
    source: Neuron
    target: Neuron

    def __repr__(self) -> str:
        return f"({self.source} -> {self.target})"


def iterate_neurons(java_neuron) -> Iterable[Neuron]:
    queue = deque([java_neuron])
    while len(queue) > 0:
        n = queue.popleft()
        yield Neuron.build(n)

        queue.extend(n.getInputs())


def iterate_edges(java_neuron) -> Iterable[Edge]:
    queue = deque([java_neuron])
    while len(queue) > 0:
        n_right = queue.popleft()
        n_right_out = Neuron.build(n_right)

        inputs = n_right.getInputs()

        for n_left in inputs:
            yield Edge(Neuron.build(n_left), n_right_out)

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


@dataclass
class Graph:
    nodes: set[Neuron]
    edges: set[Edge]

    @classmethod
    def build(cls, raw_sample: NeuralSample) -> "Graph":
        start_neuron = raw_sample.java_sample.query.neuron
        nodes = set(iterate_neurons(start_neuron))
        edges = set(iterate_edges(start_neuron))
        return Graph(nodes, edges)

    def to_networkx(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(((n.index, n.serialize()) for n in self.nodes))
        g.add_edges_from(((e.source.index, e.target.index) for e in self.edges))

        self._fix_topological_generations(g)

        # for layer, nodes in enumerate(nx.algorithms.dag.topological_generations(g)):
        #     # `multipartite_layout` expects the layer as a node attribute, so add the
        #     # numeric layer value as a node attribute
        #     for node in nodes:
        #         g.nodes[node]["layer"] = layer

        return g

    def _fix_topological_generations(self, g: nx.DiGraph):
        layers = set((d["layer"] for _, d in g.nodes(data=True)))
        layers_map: dict[int, int] = dict()

        for i, l_original in enumerate(sorted(layers, reverse=True)):
            layers_map[l_original] = i

        for _, d in g.nodes(data=True):
            d["layer"] = layers_map[d["layer"]]

    def draw(self, labels=False):
        g = self.to_networkx()

        # Compute the multipartite_layout using the "layer" node attribute
        pos = nx.drawing.layout.multipartite_layout(g, subset_key="layer")

        fig, ax = plt.subplots()

        types = [d["type"] for _, d in g.nodes(data=True)]
        node_cmap = plt.cm.tab10
        node_color_vals, types_uniq, node_colors = get_as_colors(types, cmap=node_cmap, n=10)

        edge_out_layers = [g.nodes[t]["layer"] for s, t in g.edges]
        edge_color_vals, _ = map_to_ints(edge_out_layers)

        nx.drawing.nx_pylab.draw_networkx(
            g,
            pos=pos,
            ax=ax,
            node_color=node_color_vals,
            edge_color=edge_color_vals,
            edge_cmap=plt.cm.tab10,
            with_labels=labels,
            labels={i: d['layer'] for i, d in g.nodes(data=True)}
        )
        ax.set_title("DAG layout in topological order")
        fig.tight_layout()
        legend_elements = [
            mpl.lines.Line2D([0, 1], [0, 0], marker="o", color="w", label=t, markerfacecolor=c, markersize=15)
            for t, c in zip(types_uniq, node_colors)
        ]
        plt.legend(handles=legend_elements, loc="upper right")
        return fig


if __name__ == "__main__":
    try:
        dataset = MyMutagenesis()
        dataset.settings.compute_neuron_layer_indices = True
        built_dataset = dataset.build()

        out_dir = Path(f"./imgs/{dataset.name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        i = 107
        graph = Graph.build(built_dataset.samples[i])
        g = graph.to_networkx()
        graph.draw(labels=True)
        plt.show()
    except jpype.JException as e:
        print(e.message())
        print(e.stacktrace())
        raise e
    # plt.savefig(out_dir / f"{i}.jpg")

#     for i, sample in tqdm(enumerate(built_dataset.samples), total=len(built_dataset.samples)):
#         Graph.build(sample).draw()
#         plt.savefig(out_dir / f"{i}.jpg")
