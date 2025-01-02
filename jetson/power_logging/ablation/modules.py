"""Ablated nn modules definitions."""
import torch
import math

def cast_to_tuple_2d(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    else:
        return value


class AblatedAbstract2d(torch.nn.Module):
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]

    def __init__(self, layer: torch.nn.Module) -> None:
        super().__init__()
        self.kernel_size = cast_to_tuple_2d(layer.kernel_size)
        self.stride = cast_to_tuple_2d(layer.stride)
        self.padding = cast_to_tuple_2d(layer.padding)
        self.dilation = cast_to_tuple_2d(layer.dilation)

    def forward(self, x):
        return torch.ones(self.output_shape(x))

    def output_shape(self, x) -> tuple[int, int, int, int]:
        x_shape = x.shape
        n = x_shape[0]
        c = x_shape[1]
        round_ = math.floor
        h_out = round_(
            (x_shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = round_(
            (x_shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        return n, c, h_out, w_out

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"
            f", dilation={self.dilation}"
        )


class AblatedConv2d(AblatedAbstract2d):
    out_channels: int

    def __init__(self, layer: torch.nn.Conv2d) -> None:
        super().__init__(layer)
        self.out_channels = layer.out_channels

    def output_shape(self, x) -> tuple[int, int, int, int]:
        shape = super().output_shape(x)
        return (
            shape[0],
            self.out_channels,
            shape[2],
            shape[3],
        )

    def extra_repr(self) -> str:
        return f"out_channels={self.out_channels}, {super().extra_repr()}"


class AblatedPool2d(AblatedAbstract2d):
    pass


class AblatedAdaptivePool2d(torch.nn.Module):
    def __init__(self, layer: torch.nn.AdaptiveAvgPool2d) -> None:
        super().__init__()
        self.output_size = layer.output_size

    def forward(self, x):
        shape = x.shape
        return torch.ones((shape[0], shape[1], self.output_size[0], self.output_size[1]))

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


class AblatedLinear(torch.nn.Module):
    out_features: int

    def __init__(self, layer: torch.nn.Linear) -> None:
        super().__init__()
        self.out_features = layer.out_features

    def forward(self, x):
        shape = x.shape
        return torch.ones((shape[0], self.out_features))

    def extra_repr(self) -> str:
        return f"out_features={self.out_features}"