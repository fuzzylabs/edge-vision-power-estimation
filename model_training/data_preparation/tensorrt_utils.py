import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_core.core_schema import ValidationInfo

class TensorRTInputOutput(BaseModel):
    """TensorRT layer input and output model."""

    dimensions: list[int] = Field(validation_alias="Dimensions")

    @field_validator("dimensions")
    def check_dimensions(cls, dimensions: list[int], _: ValidationInfo) -> list[int]:
        """Check dimensions of input/output tensor."""
        if len(dimensions) not in [2, 4]:
            raise ValidationError("Tensor must have 2 or 4 dimensions")

        return dimensions


class TensorRTLayer(BaseModel):
    """TensorRT layer definition."""

    name: str = Field(validation_alias="Name")

    inputs: list[TensorRTInputOutput] = Field(validation_alias="Inputs")
    outputs: list[TensorRTInputOutput] = Field(validation_alias="Outputs")
    layer_type: str = Field(validation_alias="LayerType")
    parameter_type: str | None = Field(validation_alias="ParameterType", default=None)
    kernel: list[int] | None = Field(validation_alias="Kernel", default=None)
    pre_padding: list[int] | None = Field(validation_alias="PrePadding", default=None)
    post_padding: list[int] | None = Field(validation_alias="PostPadding", default=None)
    stride: list[int] | None = Field(validation_alias="Stride", default=None)
    dilation: list[int] | None = Field(validation_alias="Dilation", default=None)

    @field_validator("kernel", "pre_padding", "post_padding", "stride", "dilation")
    def ensure_2d(cls, val: list[int], _: ValidationInfo) -> list[int]:
        """Check dimensions of input/output tensor."""
        if len(val) not in [2, 4]:
            raise ValidationError("Tensor must have 2 or 4 dimensions")

        return val

    def get_layer_type(self) -> str:
        """Get layer type.

        Returns:
            str: Name of the layer
        """
        if self.parameter_type == "Pooling":
            return "pooling"
        elif self.parameter_type == "Convolution":
            return "convolutional"
        elif self.layer_type == "gemm":
            return "dense"
        else:
            return self.layer_type

class TensorRTEngineInfo(BaseModel):
    """TensorRT engine information model."""

    class Config:
        """Model configuration."""

        extra = "ignore"

    layers: list[TensorRTLayer] = Field(validation_alias="Layers")


def read_layers_info(path: Path) -> dict[str, TensorRTLayer]:
    """Read TensorRT engine info from file."""
    with open(path, "r") as f:
        json_content = json.load(f)
        info = TensorRTEngineInfo.model_validate(json_content)
        return {layer.name: layer for layer in info.layers}
