from collections.abc import Callable
from dataclasses import dataclass

import torch
from neuralogic.core import BuiltDataset, Template
from neuralogic.dataset import BaseDataset
from neuralogic.nn.java import NeuraLogic
from torch_geometric.data.dataset import Dataset

from compute_graph_vectorize.sources.neuralogic_settings import NeuralogicSettings


@dataclass
class BuiltDatasetInstance:
    neuralogic: NeuraLogic
    built_dataset: BuiltDataset
    pyg_data: tuple[Dataset, Callable[[], torch.nn.Module]] | None = None

    @property
    def samples(self):
        return self.built_dataset.samples


class MyDataset:
    def __init__(
        self,
        name: str,
        template: Callable[[], Template],
        dataset: BaseDataset,
        neuralogic_settings: NeuralogicSettings,
    ) -> None:
        self.name = name
        self.template = template
        self.dataset = dataset
        self._n_settings = neuralogic_settings

    def build(self, sample_run=False) -> BuiltDatasetInstance:
        neuralogic = self.template().build(self._n_settings)
        built_dataset = neuralogic.build_dataset(self.dataset)

        if sample_run:
            neuralogic(built_dataset, train=False)

        return BuiltDatasetInstance(neuralogic, built_dataset)
