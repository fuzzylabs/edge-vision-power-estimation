"""Get summary of model using PyTorch model file."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from loguru import logger


def get_layers(
    model: torch.nn.Module, name_prefix: str = ""
) -> list[Tuple[str, torch.nn.Module]]:
    """
    Recursively get all layers in a pytorch model.

    Args:
        model: the pytorch model to look for layers.
        name_prefix: Use to identify the parents layer. Defaults to "".

    Returns:
        a list of tuple containing the layer name and the layer.
    """
    children = list(model.named_children())

    if not children:
        return [(name_prefix, model)] if name_prefix else [("root", model)]
    
    result = []
    for child_name, child in children:
        full_name = f"{name_prefix}.{child_name}" if name_prefix else child_name
        result.extend(get_layers(child, full_name))

    return result

    # if len(children) == 0:
    #     result = [(name_prefix, model)]
    # else:
    #     result = []
    #     for child_name, child in children:
    #         layers = get_layers(child, name_prefix + "_" + child_name)
    #         result.extend(layers)

    # return result


def get_summary(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    summary_file_path: str = "",
) -> Dict[str, Dict[str, Any]]:
    """
    Get key information of all layers within a model.

    Args:
        model: The pytorch model
        input_shape: input size of the model
        summary_file_path: Path to save model summary as a JSON file

    Returns:
        information about the model
    """
    model_info = Dict[str, Dict[str, Any]] = {}
    test = torch.randn(*input_shape)
    hooks = []
 
    def register_hook(layer_name: str):
        def hook(module, input, output):
            model_info[layer_name] = {
                "input_shape": tuple(input[0].size()) if input else None,
                "output_shape": tuple(output.size()) if output is not None else None,
                "kernel_size": getattr(module, "kernel_size", None),
                "stride": getattr(module, "stride", None),
                "padding": getattr(module, "padding", None),
                "type": module.__class__.__name__,
            }

        return hook

    for layer_name, layer in get_layers(model):
        hooks.append(layer.register_forward_hook(register_hook(layer_name)))

    model.eval()
    with torch.no_grad():
        _ = model(test)

    for hook in hooks:
        hook.remove()

    if summary_file_path:
        Path(summary_file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file_path, "w") as file:
            json.dump(
                model_info, file, indent=4, separators=(",", ": "), ensure_ascii=False
            )
        print(f"Saved summary json to {summary_file_path}")

    return model_info
