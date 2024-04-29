import torch
import torch.nn.functional as F
from neuralogic.core.builder.builder import NeuralSample
from torch_geometric.loader import DataLoader

from lib.benchmarks.runnables.runnable import Runnable
from lib.benchmarks.utils.timer import Timer
from lib.datasets.dataset import BuiltDatasetInstance


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
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def forward_pass(self):
        return self.model(self.data.x, self.data.edge_index, self.data.batch)

    def measure_forward_pass_epoch(self, timer: Timer):
        assert timer.device == self.device

        with timer:
            self.model(self.data.x, self.data.edge_index, self.data.batch)

    def measure_forward_and_backward_pass_epoch(
        self,
        forward_timer: Timer,
        backward_timer: Timer,
        combined_timer: Timer,
    ):
        assert forward_timer.device == self.device
        assert backward_timer.device == self.device
        assert combined_timer.device == self.device

        y = self.data.y.to(torch.get_default_dtype()).squeeze()

        self.optimizer.zero_grad()
        with combined_timer:
            with forward_timer:
                out = self.model(self.data.x, self.data.edge_index, self.data.batch)
            loss = F.mse_loss(out.squeeze(), y)
            with backward_timer:
                loss.backward()
            self.optimizer.step()

    @property
    def device(self):
        return self._device
