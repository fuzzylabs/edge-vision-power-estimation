import torch
from typing import Any
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
    
model = load_model("convnext_base", "pytorch/vision")
    
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

    if len(children) == 0:
        result = [(name_prefix, model)]
    else:
        result = []
        for child_name, child in children:
            layers = get_layers(child, name_prefix + "_" + child_name)
            result.extend(layers)
    
    return result