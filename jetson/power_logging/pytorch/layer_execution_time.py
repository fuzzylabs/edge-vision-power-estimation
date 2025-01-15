"""Script."""

import torch
from typing import Any
import time
from functools import partial
import datetime
import json


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
    try:
        return torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
    except:
        raise ValueError(
            f"Model name: {model_name} is most likely incorrect. "
            "Please refer https://pytorch.org/hub/ to get model name."
        )

def get_layers(model: torch.nn.Module, name_prefix: str="") -> list[tuple[str, torch.nn.Module]]:
    """
    Recursively get all layers in a pytorch model.

    Args:
        model: the pytorch model to look for layers.
        name_prefix: Use to identify the parents layer. Defaults to "".

    Returns:
        a list of tuple containing the layer name and the layer.
    """
    children = list(model.named_children())

    if len(children) == 0: # No child
        result = [(name_prefix, model)]
    else:
        # If have children, iterate over each child.
        result = []
        for child_name, child in children:
            # Recursively call get_layers on the child, appending the current
            # child's name to the name_prefix.
            layers = get_layers(child, name_prefix + "_" + child_name)
            result.extend(layers)
    
    return result


def define_and_register_hooks(model) -> None:
    """
    Define and register layer hooks.
    """
    layer_time_dict = {}

    for layer_name, layer in get_layers(model):
        layer.register_forward_pre_hook(partial(layer_time_pre_hook, layer_time_dict, layer_name))
        layer.register_forward_hook(partial(layer_time_hook, layer_time_dict, layer_name))
    
    return layer_time_dict

def layer_time_pre_hook(layer_time_dict, layer_name, module, input) -> None:
    """
    Forward pass pre-hook.

    Args:
        layer_time_dict: dictionary to save hook function output.
        layer_name: the layer to register hook.
        module: the module to register hook.
        input: tuple containing the input arguments to module's forward method.
    """
    layer_time_dict[layer_name] = (time.time(), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def layer_time_hook(layer_time_dict, layer_name, module, input, output) -> None:
    """
    Forward pass hook.

    Args:
        layer_time_dict: dictionary to save hook function output.
        layer_name: the layer to register hook.
        module: the module to register hook.
        input: tuple containing the input arguments to module's forward method.
        output: the output tensor from the forward method.
    """
    layer_time_dict[layer_name] =  (time.time() - layer_time_dict[layer_name][0], layer_time_dict[layer_name][1]) 


def get_layer_execution_time(model_name, input_shape, num_inference_cycles) -> None:
    """
    Benchmark function.

    This function will output a json file containing the recorded execution time for all layer for all inference cycles ran.

    Args:
        model_name: the name of the model to run benchmark on.
        input_shape: shape of the input tensor for inference.
        num_inference_cycles: number of cycles to run inference.
    """
    all_cycle_measurements = {}
    model = load_model(model_name)
    x = torch.randn(input_shape)

    layer_time_dict = define_and_register_hooks(model)

    for i in range(num_inference_cycles):
        model(x)
        all_cycle_measurements[f"cycle_{i}"] = layer_time_dict

    with open(f"{datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S')}_{model_name}_inference_trace_{num_inference_cycles}_cycles.json", "w") as f:
        json.dump(all_cycle_measurements, f)

get_layer_execution_time("resnet18", ((1,3,224,224)), 2)
