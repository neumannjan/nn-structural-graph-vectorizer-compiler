from compute_graph_vectorize.engines.torch.model import LayeredInputType

import torch


class LinearModule(torch.nn.Module):
    def __init__(self, retrieve_weights: torch.nn.Module) -> None:
        super().__init__()
        self.retrieve_weights = retrieve_weights

    def forward(self, x: torch.Tensor, inputs: LayeredInputType):
        w = self.retrieve_weights(inputs)
        y = w @ x
        return y
