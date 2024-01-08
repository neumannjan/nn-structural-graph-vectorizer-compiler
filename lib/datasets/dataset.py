from neuralogic.core import BuiltDataset, Settings, Template
from neuralogic.dataset import BaseDataset


class MyDataset:
    def __init__(self, name: str, template: Template, dataset: BaseDataset) -> None:
        self.name = name
        self.template = template
        self.dataset = dataset
        self.settings = Settings()

    def build(self) -> BuiltDataset:
        return self.template.build(self.settings).build_dataset(self.dataset)
