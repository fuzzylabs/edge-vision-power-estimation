import torch
import json
import argparse
from typing import Any 
from model_utils import load_model, get_layers

def get_layer_info(model, input_shape):
    """
    Get key information of all layers within a model.

    Args:
        model: The pytorch model
        input_shape: input size of the model

    Returns:
        information about the model
    """
    model_info = {}
    test = torch.randn(*input_shape)
    hooks = []

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

    for layer_name, layer in get_layers(model):
        hooks.append(layer.register_forward_hook(register_hook(layer_name)))
    
    model.eval()
    with torch.no_grad():
        _ = model(test)

    for hook in hooks:
        hook.remove()
            
    return model_info


def run(args):
    model = load_model(args.model, args.model_repo)
    layer_info = get_layer_info(model, args.input_shape)

    print(json.dumps(layer_info, indent=4, separators=(",", ": "), ensure_ascii=False))

    output = "model_summary.json"
    with open(output, "w") as file:
        json.dump(layer_info, file, indent=4, separators=(",", ": "), ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Save Model Summary",
        description="Save a summary of the model after benchmarking.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="Specify name of pretrained CNN mode from PyTorch Hub."
        "For more information on PyTorch Hub visit: "
        "https://pytorch.org/hub/research-models",
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default="pytorch/vision:v0.10.0",
        help="Specify path and version to model repository from PyTorch Hub."
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs="+",
        default=[1, 3, 224, 224],
        help="Input shape BCHW",
    )

    args = parser.parse_args()
    run(args)