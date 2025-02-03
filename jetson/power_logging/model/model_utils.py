from typing import Any

import torch
from ultralytics import YOLO

from model.lenet import LeNet

# from model.pytorch_quantize import quantized_pt_model


def load_model(model_name: str) -> Any:
    """Load model from Pytorch Hub.

    Args:
        model_name: Name of model.
            It should be same as that in Pytorch Hub.

    Raises:
        ValueError: If loading model fails from PyTorch Hub

    Returns:
        PyTorch model
    """
    if "quant" in model_name:
        # return quantized_pt_model(model_name, "coco.yaml", "datasets/coco/val2017.txt")
        return ""
    else:
        return YOLO(f"{model_name}.pt")


def load_onnx_model(model_name: str) -> Any:
    """Load model from the file.

    Args:
        model_name: Name of model.

    Returns:
        ONNX model
    """
    return YOLO(f"{model_name}.onnx")


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
