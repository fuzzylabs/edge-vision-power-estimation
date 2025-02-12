from typing import Any

import torch

from model.lenet import LeNet


def load_model(
    model_name: str,
    in_channels: int = 3,
    kernel_size: int = 1,
    stride_size: int = 1,
    out_channels: int = 1,
) -> Any:
    """Load model using ultralytics library.

    Args:
        model_name: Name of model.

    Returns:
        YOLO model
    """
    if "quant.pt" in model_name:
        from model.pytorch_quantize import quantized_pt_model

        return quantized_pt_model(model_name, "coco.yaml", "datasets/coco/val2017.txt")
    elif model_name == "single_conv_layer":
        return torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride_size,
        )
    else:
        from ultralytics import YOLO

        return YOLO(f"{model_name}")


def get_layers(
    model: torch.nn.Module, name_prefix: str = ""
) -> list[tuple[str, torch.nn.Module]]:
    """
    Recursively get all layers in a pytorch model.

    Args:
        model: the pytorch model to look for layers.
        name_prefix: Use to identify the parents layer. Defaults to "".

    Returns:
        a list of tuple containing the layer name and the layer.
    """
    children = list(model.named_children())

    if len(children) == 0:
        result = [(name_prefix, model)]
    else:
        result = []
        for child_name, child in children:
            layers = get_layers(child, name_prefix + "_" + child_name)
            result.extend(layers)

    return result
