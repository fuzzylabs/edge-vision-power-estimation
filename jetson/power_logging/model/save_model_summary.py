import torch
from typing import Any 
# from torchsummary import summary
from torchinfo import summary
# from model.lenet import LeNet


def load_model(model_name: str, model_repo: str) -> Any:
    """Load model from Pytorch Hub.

    Args:
        model_name: Name of model.
            It should be same as that in Pytorch Hub.

    Raises:
        ValueError: If loading model fails from PyTorch Hub

    Returns:
        PyTorch model
    """
    # if model_name == "lenet":
    #     return LeNet()
    if model_name == "fcn_resnet50":
        return torch.hub.load(model_repo, model_name, pretrained=True)
    try:
        return torch.hub.load(model_repo, model_name)
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

def get_layer_info(model):
    model_info = {}
    for layer_name, layer in get_layers(model):
        model_info[layer_name] = {
            "kernal_size": layer.kernel_size if hasattr(layer, "kernel_size") else None,
            "stride": layer.stride if hasattr(layer, "stride") else None,
            "padding": layer.padding if hasattr(layer, "padding") else None,
            "type": type(layer)
        }

    return model_info