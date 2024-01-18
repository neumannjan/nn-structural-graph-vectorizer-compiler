import torch
from tqdm.auto import tqdm

from lib.utils import expand_diag, value_to_tensor


class Linear(torch.nn.Module):
    def __init__(
        self,
        layer_neurons: list,
        assume_all_weights_same: bool,
    ) -> None:
        super().__init__()

        if assume_all_weights_same:
            layer_neurons = layer_neurons[:1]

        self.weights_map = torch.nn.ParameterDict()
        self.all_weights = []

        self.weight_indices = []
        for weight in tqdm([w for n in layer_neurons for w in n.getWeights()], desc="Weights"):
            self.weight_indices += [weight.index]

            if str(weight.index) in self.weights_map:
                w_tensor = self.weights_map[str(weight.index)]
            else:
                w_tensor = value_to_tensor(weight.value)

                if weight.isLearnable:
                    w_tensor = torch.nn.Parameter(w_tensor)

                self.weights_map[str(weight.index)] = w_tensor

            self.all_weights += [w_tensor]

        # ensure all weights are square matrices, vectors, or scalars
        for v in self.all_weights:
            assert v.dim() <= 2
            if torch.squeeze(v).dim() == 2:
                assert v.shape[0] == v.shape[1]

        # weight info
        w_max_dim: int = max((v.dim() for v in self.all_weights))
        self.w_shape_hull: list[int] = [
            max((v.shape[i] for v in self.all_weights if i < v.dim())) for i in range(w_max_dim)
        ]
        self.expand_diagonal: bool = w_max_dim == 2 and self.w_shape_hull[0] == self.w_shape_hull[1]

    def forward(self, input_values: torch.Tensor):
        if self.expand_diagonal:
            w = [expand_diag(v, self.w_shape_hull[0]) for v in self.all_weights]
        else:
            w = [v.expand(self.w_shape_hull) for v in self.all_weights]

        w = torch.stack(w)
        y = w @ input_values
        return y
