import torch
import torch.nn as nn

def zero_keep_pruning(model, threshold=0.0):
    """
    Apply ZKP (Zero-Keep Pruning) to a PyTorch Model

    Args:
        model: PyTorch Model to prune
        threshold: Threshold in which models are pruned (default is 0)

    Returns:
        pruned_model: Model with the pruned weights
        pruning_masks: Dictionary of binary masks for each layer
    """
    pruning_masks = {}

    for name, param in model.named_parameters():
        if "weight" in name:
            # Will mask the weights above the threshold
            mask = (param.abs() > threshold).float()
            pruning_masks[name] = mask

            param.data *= mask
    return model, pruning_masks