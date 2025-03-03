import json
import torch
import torch.nn as nn
from typing import Dict
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_core.core_schema import ValidationInfo

POOLING_LAYER_NAMES = [
    "AvgPool2d",
    "MaxPool2d",
    "AdaptiveAvgPool2d",
    "AdaptiveMaxPool2d",
]

class PytorchLayer(BaseModel):
    """Pytorch layer definition."""

    input_shape: list[int] = Field()
    output_shape: list[int] = Field()
    layer_type: str = Field(validation_alias="type")
    kernel_size: list[int] | int | None = Field(default=None)
    padding: list[int] | int | None = Field(default=None)
    stride: list[int] | int | None = Field(default=None)

    @field_validator("kernel_size", "padding", "stride")
    def ensure_2d(cls, val: list[int], _: ValidationInfo) -> list[int]:
        """Check dimensions of input/output tensor."""
        if val is None:
            return None
        if isinstance(val, int):
            return [val, val]
        if len(val) not in [2]:
            raise ValidationError("Tensor must have 2 dimensions")

        return val

    def get_layer_type(self) -> str:
        """Get layer type.

        Returns:
            str: Name of the layer
        """
        if self.layer_type in POOLING_LAYER_NAMES:
            return "pooling"
        elif self.layer_type == "Conv2d":
            return "convolutional"
        elif self.layer_type == "Linear":
            return "dense"
        else:
            return self.layer_type

PytorchModelSummary = dict[str, PytorchLayer]

def read_layers_info(path: Path) -> PytorchModelSummary:
    """Read Pytorch model summary from file."""
    model_summary = {}
    with open(path, "r") as f:
        json_content = json.load(f)
        for layer_name, layer_dict in json_content.items():
            layer = PytorchLayer.model_validate(layer_dict)
            model_summary[layer_name] = layer
        return model_summary

def read_layers_info_from_model(model: nn.Module, input_shape=(1, 3, 224, 224)) -> PytorchModelSummary:
    """
    Reading from a model specifically
    """

    model_summary: PytorchModelSummary = {}

    for name, module in model.named_modules():
        if name == "":
            continue

        layer_type = module.__class__.__name__

        kernel_size = None
        padding = None
        stride = None

        if hasattr(module, "kernel_size"):
            kernel_size = module.kernel_size
        if hasattr(module, "padding"):
            padding = module.padding
        if hasattr(module, "stride"):
            stride = module.stride

        output_shape_ = [0, 0, 0]

        layer_obj = PytorchLayer(
            input_shape=input_shape,
            output_shape=output_shape_,
            layer_type=layer_type,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )

        model_summary[name] = layer_obj

    return model_summary
