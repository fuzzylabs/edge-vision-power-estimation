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

    def  __init__(self, probe_output: torch.Tensor) -> None:
        super().__init__()
        self.shape = probe_output.shape
        self.dtype = probe_output.dtype
        self.device = probe_output.device


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rand(self.shape, device=self.device, dtype=self.dtype)
