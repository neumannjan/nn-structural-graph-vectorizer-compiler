import torch


class ReshapeWithPeriod(torch.nn.Module):
    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = period

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape([-1, self.period, *x.shape[1:]])


class ViewWithPeriod(torch.nn.Module):
    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = period

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view([-1, self.period, *x.shape[1:]])


class Repeat(torch.nn.Module):
    def __init__(self, repeats: int, total_length: int) -> None:
        super().__init__()
        self.repeats = repeats
        self.total_length = total_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat(self.repeats, *([1] * (x.dim() - 1)))
        x = x[: self.total_length]
        return x


class Unsqueeze(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(self.dim)


class Expand0(torch.nn.Module):
    def __init__(self, i: int) -> None:
        super().__init__()
        self.i = i

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.expand(self.i, *x.shape[1:])
        return x
