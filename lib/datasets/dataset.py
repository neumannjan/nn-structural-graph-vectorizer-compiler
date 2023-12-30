from neuralogic.core import BuiltDataset, Settings, Template
from neuralogic.dataset import BaseDataset


class MyDataset:
    def __init__(self, name: str, template: Template, dataset: BaseDataset) -> None:
        self.name = name
        self.template = template
        self.dataset = dataset

    def build(self) -> BuiltDataset:
        return self.template.build(Settings()).build_dataset(self.dataset)
