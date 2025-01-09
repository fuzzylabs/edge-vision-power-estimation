"""Functions for performning ablation on neural networks."""
from types import MethodType

import torch

from ablation.modules import AblatedModule

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
        self._probe_output = output_tensor
        return output_tensor
    return probe

def ablate_by_key(model: torch.nn.Module, key: list[str], x: torch.Tensor) -> torch.nn.Module:
    module = model
    while len(key) > 1:
        module = module._modules[key[0]]
        key = key[1:]

    ablated_module = module._modules[key[0]]

    # Probe
    real_forward = ablated_module.forward
    ablated_module.forward = MethodType(get_probe(real_forward), ablated_module)
    _ = model(x)

    # Ablate
    module._modules[key[0]] = AblatedModule(ablated_module._probe_output)

    return model