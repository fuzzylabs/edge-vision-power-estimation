"""Functions for performning ablation on neural networks."""
from types import MethodType

import torch

from ablation.modules import AblatedConv2d, AblatedPool2d, AblatedLinear, AblatedAdaptivePool2d

layer_types_for_ablation = [
    "Linear",
    "MaxPool2d",
    "Conv2d",
    "AdaptiveAvgPool2d",
]

def get_layers_for_ablation(model: torch.nn.Module) -> list[list[str]]:
    if len(model._modules) == 0:
        return

    results = []

    for key, module in model._modules.items():
        if len(module._modules) == 0:
            module_name = module._get_name()
            if module_name in layer_types_for_ablation:
                results.append([key])
        else:
            detected_submodules = get_layers_for_ablation(module)
            for submodule in detected_submodules:
                results.append([key] + submodule)
    return results

def get_probe(method):
    def probe(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output_tensor = method(input_tensor)
        self._zero_tensor = torch.zeros(output_tensor.shape, dtype=output_tensor.dtype, device=output_tensor.device)
        return output_tensor
    return probe

def ablated_forward(self, x: torch.Tensor) -> torch.Tensor:
    return self._zero_tensor

def ablate(layer: torch.nn.Module) -> torch.nn.Module:
    layer_name = layer._get_name()
    if layer_name in ["Conv2d"]:
        return AblatedConv2d(layer)
    elif layer_name in ["MaxPool2d"]:
        return AblatedPool2d(layer)
    elif layer_name in ["Linear"]:
        return AblatedLinear(layer)
    elif layer_name in ["AdaptiveAvgPool2d"]:
        return AblatedAdaptivePool2d(layer)


def ablate_by_key(model: torch.nn.Module, key: list[str], x: torch.Tensor) -> torch.nn.Module:
    module = model
    while len(key) > 1:
        module = module._modules[key[0]]
        key = key[1:]
    module = module._modules[key[0]]

    # Probe
    real_forward = module.forward
    module.forward = MethodType(get_probe(real_forward), module)
    _ = model(x)

    # Ablate
    module.forward = MethodType(ablated_forward, module)

    return model