from dataclasses import dataclass
from typing import NamedTuple

from neuralogic.core import BuiltDataset, Settings, Template
from neuralogic.dataset import BaseDataset
from neuralogic.nn.java import NeuraLogic


class BuiltDatasetInstance(NamedTuple):
    neuralogic: NeuraLogic
    built_dataset: BuiltDataset

    @property
    def samples(self):
        return self.built_dataset.samples


class MyDataset:
    def __init__(self, name: str, template: Template, dataset: BaseDataset) -> None:
        self.name = name
        self.template = template
        self.dataset = dataset
        self.settings = Settings(compute_neuron_layer_indices=True)

    def build(self, sample_run=False) -> BuiltDatasetInstance:
        neuralogic = self.template.build(self.settings)
        built_dataset = neuralogic.build_dataset(self.dataset)

        if sample_run:
            neuralogic(built_dataset, train=False)

        return BuiltDatasetInstance(neuralogic, built_dataset)
