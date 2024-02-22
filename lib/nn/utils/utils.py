import torch


class ViewWithPeriod(torch.nn.Module):
    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = period

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view([-1, self.period, *x.shape[1:]])

    def extra_repr(self) -> str:
        return f"period={self.period}"


class Repeat(torch.nn.Module):
    def __init__(self, repeats: int, total_length: int) -> None:
        super().__init__()
        self.repeats = repeats
        self.total_length = total_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat(self.repeats, *([1] * (x.dim() - 1)))
        x = x[: self.total_length]
        return x

    def extra_repr(self) -> str:
        return f"repeats={self.repeats}, total_length={self.total_length}"
