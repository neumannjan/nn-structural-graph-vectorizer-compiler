from neuralogic.core import BuiltDataset, Settings, Template
from neuralogic.dataset import BaseDataset


class MyDataset:
    def __init__(self, name: str, template: Template, dataset: BaseDataset) -> None:
        self.name = name
        self.template = template
        self.dataset = dataset
        self.settings = Settings()

    def build(self, sample_run=False) -> BuiltDataset:
        neuralogic = self.template.build(self.settings)
        built_dataset = neuralogic.build_dataset(self.dataset)
        if sample_run:
            neuralogic(built_dataset, train=False)
        return built_dataset
