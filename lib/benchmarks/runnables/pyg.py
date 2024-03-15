from neuralogic.core.builder.builder import NeuralSample
from torch_geometric.loader import DataLoader

from lib.benchmarks.runnables.runnable import Runnable
from lib.datasets.dataset import BuiltDatasetInstance
from lib.datasets.tu_molecular import MyTUDataset


class PytorchGeometricRunnable(Runnable):
    def __init__(self, device: str) -> None:
        self._device = device

    def initialize(self, dataset: BuiltDatasetInstance, samples: list[NeuralSample] | None = None):
        if dataset.pyg_data is None:
            raise NotImplementedError(
                f"Missing pytorch geometric implementation for dataset {dataset.__class__.__name__}: {dataset}"
            )

        pyg_dataset, module_provider = dataset.pyg_data
        data_loader = DataLoader(pyg_dataset, batch_size=len(pyg_dataset))

        data = list(iter(data_loader))
        assert len(data) == 1
        self.data = data[0].to(self.device)

        self.model = module_provider()
        self.model.to(self.device)

    def forward_pass(self):
        return self.model(self.data.x, self.data.edge_index, self.data.batch)

    @property
    def device(self):
        return self._device


if __name__ == "__main__":
    _r = PytorchGeometricRunnable(device='cpu')
    _r.initialize(MyTUDataset(source='mutag', template='gcn').build())
    _r.forward_pass()
