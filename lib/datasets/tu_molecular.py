from pathlib import Path
from typing import Literal, Protocol

from neuralogic.core import Aggregation, R, Template, Transformation, V
from neuralogic.dataset import Data, TensorDataset
from neuralogic.nn.init import Glorot, Uniform
from torch_geometric.datasets import TUDataset

from lib.datasets.dataset import BuiltDatasetInstance, MyDataset
from lib.datasets.pyg.tu_molecular import build_pyg_module
from lib.sources.neuralogic_settings import NeuralogicSettings


class _TemplateProtocol(Protocol):
    def __call__(self, activation: Transformation, num_features: int, output_size: int, dim: int = 10) -> Template: ...


def _gcn(activation: Transformation, num_features: int, output_size: int, dim: int = 10):
    template = Template()

    # template += (R.atom_embed(V.X)[dim, num_features] <= R.node_feature(V.X)) | [Transformation.IDENTITY]
    # template += R.atom_embed / 1 | [Transformation.IDENTITY]

    template += (R.l1_embed(V.X) <= (R.node_feature(V.Y)[dim, num_features], R._edge(V.Y, V.X))) | [
        Aggregation.SUM,
        Transformation.IDENTITY,
    ]
    template += R.l1_embed / 1 | [Transformation.RELU]

    template += (R.l2_embed(V.X) <= (R.l1_embed(V.Y)[dim, dim], R._edge(V.Y, V.X))) | [
        Aggregation.SUM,
        Transformation.IDENTITY,
    ]
    template += R.l2_embed / 1 | [Transformation.IDENTITY]

    template += (R.predict[output_size, dim] <= R.l2_embed(V.X)) | [Aggregation.AVG, Transformation.IDENTITY]
    template += R.predict / 0 | [activation]

    return template


def _gin(activation: Transformation, num_features: int, output_size: int, dim: int = 10):
    template = Template()

    template += (R.atom_embed(V.X)[dim, num_features] <= R.node_feature(V.X)) | [Transformation.IDENTITY]
    template += R.atom_embed / 1 | [Transformation.IDENTITY]

    template += (R.l1_embed(V.X) <= (R.atom_embed(V.Y), R._edge(V.Y, V.X))) | [Aggregation.SUM, Transformation.IDENTITY]
    template += (R.l1_embed(V.X) <= R.atom_embed(V.X)) | [Transformation.IDENTITY]
    template += R.l1_embed / 1 | [Transformation.IDENTITY]

    template += (R.l1_mlp_embed(V.X)[dim, dim] <= R.l1_embed(V.X)[dim, dim]) | [Transformation.RELU]
    template += R.l1_mlp_embed / 1 | [Transformation.RELU]

    # --
    template += (R.l2_embed(V.X) <= (R.l1_mlp_embed(V.Y), R._edge(V.Y, V.X))) | [
        Aggregation.SUM,
        Transformation.IDENTITY,
    ]
    template += (R.l2_embed(V.X) <= R.l1_mlp_embed(V.X)) | [Transformation.IDENTITY]
    template += R.l2_embed / 1 | [Transformation.IDENTITY]

    template += (R.l2_mlp_embed(V.X)[dim, dim] <= R.l2_embed(V.X)[dim, dim]) | [Transformation.RELU]
    template += R.l2_mlp_embed / 1 | [Transformation.RELU]

    # --
    template += (R.l3_embed(V.X) <= (R.l2_mlp_embed(V.Y), R._edge(V.Y, V.X))) | [
        Aggregation.SUM,
        Transformation.IDENTITY,
    ]
    template += (R.l3_embed(V.X) <= R.l2_mlp_embed(V.X)) | [Transformation.IDENTITY]
    template += R.l3_embed / 1 | [Transformation.IDENTITY]

    template += (R.l3_mlp_embed(V.X)[dim, dim] <= R.l3_embed(V.X)[dim, dim]) | [Transformation.RELU]
    template += R.l3_mlp_embed / 1 | [Transformation.RELU]

    # --
    template += (R.l4_embed(V.X) <= (R.l3_mlp_embed(V.Y), R._edge(V.Y, V.X))) | [
        Aggregation.SUM,
        Transformation.IDENTITY,
    ]
    template += (R.l4_embed(V.X) <= R.l3_mlp_embed(V.X)) | [Transformation.IDENTITY]
    template += R.l4_embed / 1 | [Transformation.IDENTITY]

    template += (R.l4_mlp_embed(V.X)[dim, dim] <= R.l4_embed(V.X)[dim, dim]) | [Transformation.RELU]
    template += R.l4_mlp_embed / 1 | [Transformation.RELU]

    # --
    template += (R.l5_embed(V.X) <= (R.l4_mlp_embed(V.Y), R._edge(V.Y, V.X))) | [
        Aggregation.SUM,
        Transformation.IDENTITY,
    ]
    template += (R.l5_embed(V.X) <= R.l4_mlp_embed(V.X)) | [Transformation.IDENTITY]
    template += R.l5_embed / 1 | [Transformation.IDENTITY]

    template += (R.l5_mlp_embed(V.X)[dim, dim] <= R.l5_embed(V.X)[dim, dim]) | [Transformation.RELU]
    template += R.l5_mlp_embed / 1 | [Transformation.RELU]

    template += (R.predict[output_size, dim] <= R.l1_mlp_embed(V.X)) | [Aggregation.AVG, Transformation.IDENTITY]
    template += (R.predict[output_size, dim] <= R.l2_mlp_embed(V.X)) | [Aggregation.AVG, Transformation.IDENTITY]
    template += (R.predict[output_size, dim] <= R.l3_mlp_embed(V.X)) | [Aggregation.AVG, Transformation.IDENTITY]
    template += (R.predict[output_size, dim] <= R.l4_mlp_embed(V.X)) | [Aggregation.AVG, Transformation.IDENTITY]
    template += (R.predict[output_size, dim] <= R.l5_mlp_embed(V.X)) | [Aggregation.AVG, Transformation.IDENTITY]

    template += R.predict / 0 | [activation]

    return template


def _gsage(activation: Transformation, num_features: int, output_size: int, dim: int = 10):
    template = Template()

    template += (R.atom_embed(V.X)[dim, num_features] <= R.node_feature(V.X)) | [Transformation.IDENTITY]
    template += R.atom_embed / 1 | [Transformation.IDENTITY]

    template += (R.l1_embed(V.X)[dim, dim] <= R.atom_embed(V.X)) | [Transformation.IDENTITY]
    template += (R.l1_embed(V.X)[dim, dim] <= (R.atom_embed(V.Y), R._edge(V.Y, V.X))) | [
        Aggregation.AVG,
        Transformation.IDENTITY,  # TODO: revert to IDENTITY
    ]
    template += R.l1_embed / 1 | [Transformation.RELU]

    template += (R.l2_embed(V.X)[dim, dim] <= R.l1_embed(V.X)) | [Transformation.IDENTITY]
    template += (R.l2_embed(V.X)[dim, dim] <= (R.l1_embed(V.Y), R._edge(V.Y, V.X))) | [
        Aggregation.AVG,
        Transformation.IDENTITY,
    ]
    template += R.l2_embed / 1 | [Transformation.IDENTITY]

    template += (R.predict[output_size, dim] <= R.l2_embed(V.X)) | [Aggregation.AVG, Transformation.IDENTITY]
    template += R.predict / 0 | [activation]

    return template


TUDatasetTemplate = Literal["gcn", "gin", "gsage"]

_TEMPLATE_MAP: dict[TUDatasetTemplate, _TemplateProtocol] = {
    "gcn": _gcn,
    "gin": _gin,
    "gsage": _gsage,
}

TUDatasetSource = Literal["mutag", "enzymes", "proteins", "collab", "imdb-binary", "reddit-binary"]


class MyTUDataset(MyDataset):
    def __init__(self, settings: NeuralogicSettings, source: TUDatasetSource, template: TUDatasetTemplate) -> None:
        self.source_name = source
        self.template_name = template
        root = Path("./datasets")
        root.mkdir(exist_ok=True, parents=True)

        pyg_dataset = TUDataset(root=str(root), name=source.upper())

        num_node_features = pyg_dataset.num_node_features
        output_size = 1
        dim = 10

        dataset = TensorDataset(
            data=[Data.from_pyg(data)[0] for data in pyg_dataset], number_of_classes=num_node_features
        )

        def _build_template():
            return _TEMPLATE_MAP[template](
                activation=Transformation.SIGMOID, num_features=num_node_features, output_size=output_size, dim=dim
            )

        match settings.initializer:
            case Uniform(scale=2):
                pass
            case Glorot(scale=2):
                pass
            case _:
                assert False

        settings.initializer = Glorot()

        super().__init__(f"TU_{source}", _build_template, dataset, settings)
        self.pyg_dataset = pyg_dataset
        self.pyg_module_provider = lambda: build_pyg_module(
            template=template, activation="sigmoid", output_size=output_size, num_features=num_node_features, dim=dim
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(source={self.source_name}, template={self.template_name})"

    def build(self, sample_run=False) -> BuiltDatasetInstance:
        out = super().build(sample_run)

        return BuiltDatasetInstance(
            neuralogic=out.neuralogic,
            built_dataset=out.built_dataset,
            pyg_data=(self.pyg_dataset, self.pyg_module_provider),
        )
