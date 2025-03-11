"""Get summary of model using PyTorch model file."""

import json
from pathlib import Path

import torch
import torch.nn.quantized as quantized_nn
from torch.nn.intrinsic.quantized import ConvReLU2d



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
    if not hasattr(model, "_modules") or isinstance(model, (quantized_nn.Conv2d, ConvReLU2d, quantized_nn.Linear, quantized_nn.BatchNorm2d)):
        return []
    
    try:
        children = list(model.named_children())
    except AttributeError:
        return [(name_prefix, model)]

    if len(children) == 0:
        return [(name_prefix, model)]
    
    result = []
    for child_name, child in children:
        layers = get_layers(child, name_prefix + "_" + child_name)
        result.extend(layers)

    return result

def is_quantized_model(model: torch.nn.Module) -> bool:
    try:
        return any(
            isinstance(layer, (quantized_nn.Conv2d, ConvReLU2d, quantized_nn.Linear))
            for _, layer in model.named_modules()
        )
    except AttributeError:
        return False

def get_summary(
    model: torch.nn.Module,
    input_shape: tuple = (1, 3, 224, 224),
    summary_file_path: str = "",
):
    """
    Get key information of all layers within a model.

    Args:
        model: The pytorch model
        input_shape: input size of the model
        summary_file_path: Path to save model summary as a JSON file

    Returns:
        information about the model
    """
    model_info = {}
    test = torch.randn(*input_shape)
    hooks = []

    if is_quantized_model(model):
        print("Detected quantized model...")
        for layer in model.children():
            try:
                model_info[layer.__class__.__name__] = {
                    "type": layer.__class__.__name__,
                    "kernel_size": getattr(layer, "kernel_size", None),
                    "stride": getattr(layer, "stride", None),
                    "padding": getattr(layer, "padding", None),
                    "input_shape": [1, 3, 224, 224],
                    "output_shape": [1, 3, 224, 224],
                }
            except AttributeError:
                print(f"Skipping layer {layer.__class__.__name__} as it misses attribute...")
                continue
        return model_info

    def register_hook(layer_name):
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

    valid_layers = []
    for layer_name, layer in get_layers(model):
        if not hasattr(layer, "register_forward_hook"):
            continue
        valid_layers.append((layer_name, layer))

    if not valid_layers:
        raise ValueError("No valid layers found...")
    
    for layer_name, layer in valid_layers:
        hooks.append(layer.register_forward_hook(register_hook(layer_name)))

    try:
        model.eval()
    except AttributeError:
        print("model.eval() could not  be applied")
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
