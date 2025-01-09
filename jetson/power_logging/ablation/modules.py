"""Ablated nn modules definitions."""
import torch
import torch.nn.functional as F
import math

def cast_to_tuple_2d(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    else:
        return value


class AblatedModule(torch.nn.Module):
    zero_tensor: torch.Tensor

    def  __init__(self,) -> None:
        self.zero_tensor = torch.Tensor((1, 64, 112, 112)).to(torch.float16).to(torch.device('cuda'))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.zero_tensor

    def reshape(self, x: torch.Tensor, shape) -> torch.Tensor:
        # return torch.zeros(shape, dtype=x.dtype, device=x.device)
        x = x.flatten()
        in_size = x.shape[0]
        out_size = math.prod(shape)
        if in_size > out_size:
            return x[:out_size].reshape(shape)
        else:
            return F.pad(x, (0, out_size - in_size, )).reshape(shape)

    def output_shape(self, x):
        raise NotImplementedError()


class AblatedAbstract2d(AblatedModule):
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

    def output_shape(self, x) -> tuple[int, int, int, int]:
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
    out_channels: tuple[int]

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


Pool2d = torch.nn.MaxPool2d


class AblatedPool2d(AblatedAbstract2d):
    pass


class AblatedAdaptivePool2d(AblatedModule):
    def __init__(self, layer: torch.nn.AdaptiveAvgPool2d) -> None:
        super().__init__()
        self.output_size = layer.output_size

    def output_shape(self, x):
        shape = x.shape
        return shape[0], shape[1], self.output_size[0], self.output_size[1]

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


class AblatedLinear(AblatedModule):
    out_features: int

    def __init__(self, layer: torch.nn.Linear) -> None:
        super().__init__()
        self.out_features = layer.out_features

    def output_shape(self, x):
        shape = x.shape
        return shape[0], self.out_features

    def extra_repr(self) -> str:
        return f"out_features={self.out_features}"