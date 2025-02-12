from typing import Any

import torch

from model.lenet import LeNet


def extract_params(model_name: str) -> tuple[int, int, int]:
    """Extract kernel, stride and output channels from model name.

    Args:
        model_name: Name of the model.
        E.g. "conv_k1_s1_o1"

    Returns:
        A tuple of integers corresponding to kernel, stride and output channels
    """
    params = model_name.split("_")
    kernel, stride, out_channel = None, None, None
    for part in params:
        if part.startswith("k"):
            kernel = int(part[1:])
        elif part.startswith("s"):
            stride = int(part[1:])
        elif part.startswith("o"):
            out_channel = int(part[1:])
    return kernel, stride, out_channel


def load_model(model_name: str) -> Any:
    """Load model using ultralytics library.

    Args:
        model_name: Name of model.

    Returns:
        YOLO model
    """
    if "quant.pt" in model_name:
        from model.pytorch_quantize import quantized_pt_model

        return quantized_pt_model(model_name, "coco.yaml", "datasets/coco/val2017.txt")
    elif "conv_" in model_name:
        kernel_size, stride_size, out_channels = extract_params(model_name)
        if None in [kernel_size, stride_size, out_channels]:
            raise ValueError(f"Something went wrong parsing model name: {model_name}")
        return torch.nn.Conv2d(
            in_channels=3,
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
