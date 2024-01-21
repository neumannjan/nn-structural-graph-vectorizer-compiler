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

        self.weights = torch.nn.ParameterDict()
        self.weight_indices = []
        self.weight_indices_str = []

        for weight in tqdm([w for n in layer_neurons for w in n.getWeights()], desc="Weights"):
            w_idx = str(weight.index)

            if str(weight.index) in self.weights:
                w_tensor = self.weights[w_idx]
            else:
                w_tensor = torch.nn.Parameter(value_to_tensor(weight.value), requires_grad=weight.isLearnable())

                self.weights[w_idx] = w_tensor

            self.weight_indices += [weight.index]
            self.weight_indices_str += [w_idx]

        # ensure all weights are square matrices, vectors, or scalars
        for v in self.weights.values():
            assert v.dim() <= 2
            if torch.squeeze(v).dim() == 2:
                assert v.shape[0] == v.shape[1]

        # weight info
        w_max_dim: int = max((v.dim() for v in self.weights.values()))
        self.w_shape_hull: list[int] = [
            max((v.shape[i] for v in self.weights.values() if i < v.dim())) for i in range(w_max_dim)
        ]
        self.expand_diagonal: bool = w_max_dim == 2 and self.w_shape_hull[0] == self.w_shape_hull[1]

    def forward(self, input_values: torch.Tensor):
        with torch.profiler.record_function('LINEAR_EXPAND'):
            if self.expand_diagonal:
                ws = {idx: expand_diag(w, self.w_shape_hull[0]) for idx, w in self.weights.items()}
            else:
                ws = {idx: w.expand(self.w_shape_hull) for idx, w in self.weights.items()}

        with torch.profiler.record_function('LINEAR_STACK'):
            w = [ws[idx] for idx in self.weight_indices_str]
            w = torch.stack(w)

        with torch.profiler.record_function('LINEAR_MULT'):
            y = w @ input_values
        return y
