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

def read_layers_info_from_model(model: nn.Module, input_shape=(1, 3, 224, 224)) -> dict:
    """
    Reading from a model specifically
    """

    model_info = {}
    input = torch.randn(*input_shape)
    hooks = []

    def register_hook(layer_name):
        def hook(module, input, output):
            model_info[layer_name] = {
                "input_shape": list(input[0].size()) if input else None,
                "output_shape": list(output.size()) if output is not None else None,
                "kernel_size": getattr(module, "kernel_size", None),
                "stride": getattr(module, "stride", None),
                "padding": getattr(module, "padding", None),
                "type": module.__class__.__name__,
            }
        return hook
    
    for name, module in module.named_modules():
        if name:
            hooks.append(module.register_forward_hook(register_hook(name)))

    model.eval()
    with torch.no_grad():
        _ = model(input)

    for hook in hooks:
        hook.remove()

    return model_info

    # model_summary: PytorchModelSummary = {}

    # for name, module in model.named_modules():
    #     if name == "":
    #         continue

    #     layer_type = module.__class__.__name__

    #     kernel_size = [1, 1]
    #     padding = [0, 0]
    #     stride = [1, 1]

    #     if hasattr(module, "kernel_size") and module.kernel_size is not None:
    #         if isinstance(module.kernel_size, int):
    #             kernel_size = [module.kernel_size, module.kernel_size]
    #         elif isinstance(module.kernel_size, tuple):
    #             kernel_size = list(module.kernel_size)

    #     if hasattr(module, "padding") and module.padding is not None:
    #         if isinstance(module.padding, int):
    #             padding = [module.padding, module.padding]
    #         elif isinstance(module.padding, tuple):
    #             padding = list(module.padding)

    #     if hasattr(module, "stride") and module.stride is not None:
    #         if isinstance(module.stride, int):
    #             stride = [module.stride, module.stride]
    #         elif isinstance(module.stride, tuple):
    #             stride = list(module.stride)

    #     output_shape = [1, 64, 112, 112]

    #     layer_dict = {
    #         "input_shape": list(input_shape),  
    #         "output_shape": output_shape,      
    #         "type": layer_type,                
    #         "kernel_size": kernel_size,
    #         "padding": padding,
    #         "stride": stride,
    #     }

    #     layer_obj = PytorchLayer.model_validate(layer_dict)
    #     model_summary[name] = layer_obj

    # return model_summary
