"""Pruning utils."""
import torch
import torch.nn.utils.prune as prune


def get_layers_for_pruning(model: torch.nn.Module) -> tuple[(torch.nn.Module, str)]:
    """
    Recursively get all layers in a pytorch model.

    Args:
        model: the pytorch model to look for layers.

    Returns:
        a tuple of list containing the layer.
    """
    children = list(model.named_children())

    if len(children) == 0:
        result = [(model, "weight")]
    else:
        result = []
        for _, child in children:
            layers = get_layers_for_pruning(child)
            result.extend(layers)
    
    return tuple(result)


def global_unstructured_prune(
    model: torch.nn.Module,
    amount: float,
) -> None:
    """
    Prune the model by the amount of weight specified.

    Args:
        model: model to reference to.
        amount: sparsity level you are applying to a layer.
    """
    parameters_to_prune = get_layers_for_pruning(model)
    # We prune only conv and linear layers since maxpooling layer has no weight
    parameters_to_prune = [(layer, "weight") for layer, _ in parameters_to_prune if isinstance(layer, torch.nn.modules.conv.Conv2d) or isinstance(layer, torch.nn.modules.linear.Linear)]
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )