import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_core.core_schema import ValidationInfo

POOLING_LAYER_NAMES = [
    "AvgPool2d",
    "MaxPool2d",
    "AdaptiveAvgPool2d",
    "AdaptiveMaxPool2d",
]

class TensorRTInputOutput(BaseModel):
    """TensorRT layer input and output model."""

    dimensions: list[int] = Field(validation_alias="Dimensions")

    @field_validator("dimensions")
    def check_dimensions(cls, dimensions: list[int], _: ValidationInfo) -> list[int]:
        """Check dimensions of input/output tensor."""
        if len(dimensions) not in [2, 4]:
            raise ValidationError("Tensor must have 2 or 4 dimensions")

        return dimensions


class PytorchLayer(BaseModel):
    """Pytorch layer definition."""

    input_shape: list[int] = Field()
    output_shape: list[int] = Field()
    layer_type: str = Field(validation_alias="type")
    kernel_size: list[int] | None = Field(default=None)
    padding: list[int] | None = Field(default=None)
    stride: list[int] | None = Field(default=None)

    @field_validator("kernel_size", "padding", "stride")
    def ensure_2d(cls, val: list[int], _: ValidationInfo) -> list[int]:
        """Check dimensions of input/output tensor."""
        if val is None:
            return None
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
    """Read TensorRT engine info from file."""
    model_summary = {}
    with open(path, "r") as f:
        json_content = json.load(f)
        for layer_name, layer_dict in json_content.items():
            print("HERE")
            print(layer_dict)
            layer = PytorchLayer.model_validate(layer_dict)
            model_summary[layer_name] = layer
        return model_summary
