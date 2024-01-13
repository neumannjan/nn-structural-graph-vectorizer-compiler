import torch
from tqdm.auto import tqdm

from lib.utils import expand_diag, value_to_tensor


class Linear(torch.nn.Module):
    def __init__(
        self,
        layer_neurons: list,
        diagonal_expand: bool,
        assume_all_weights_same: bool,
    ) -> None:
        super().__init__()
        self.expand_diagonal = diagonal_expand

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

        if self.expand_diagonal:
            # ensure all weights are square matrices, vectors, or scalars
            for v in self.all_weights:
                assert v.dim() <= 2
                if torch.squeeze(v).dim() == 2:
                    assert v.shape[0] == v.shape[1]
        else:
            # ensure all weights are matrices
            for v in self.all_weights:
                assert v.dim() == 2

    def forward(self, input_values: torch.Tensor):
        w = self.all_weights
        if self.expand_diagonal:
            dim = max((torch.atleast_1d(torch.squeeze(v)).shape[0] for v in w))
            w = [expand_diag(v, dim) for v in w]
        else:
            # TODO: remove?
            w_shape_hull = [
                max((v.shape[0] for v in w)),
                max((v.shape[1] for v in w)),
            ]
            w = [v.expand(w_shape_hull) for v in w]

        w = torch.stack(w)
        y = w @ input_values
        return y


