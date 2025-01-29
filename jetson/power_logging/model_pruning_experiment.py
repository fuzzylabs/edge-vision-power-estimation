"""Experiment on model pruning using pytorch."""

import torch
import torch.nn.utils.prune as prune

from model.model_utils import load_model


def get_layers_for_pruning(model: torch.nn.Module) -> tuple[(torch.nn.Module, str)]:
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
        result = [(model, "weight")]
    else:
        result = []
        for _, child in children:
            layers = get_layers_for_pruning(child)
            result.extend(layers)
    
    return tuple(result)

# Lenet experiment
# model = load_model("lenet", "pytorch/vision:v0.16.0")

# parameters_to_prune = (
#     (model.feat.conv1, 'weight'),
#     (model.feat.conv2, 'weight'),
#     (model.classifer.fc1, 'weight'),
#     (model.classifer.fc2, 'weight'),
#     (model.classifer.fc3, 'weight'),
# )

# prune.global_unstructured(
#     parameters_to_prune,
#     pruning_method=prune.L1Unstructured,
#     amount=0.3,
# )

# print(
#     "Sparsity in conv1.weight: {:.2f}%".format(
#         100. * float(torch.sum(model.feat.conv1.weight == 0))
#         / float(model.feat.conv1.weight.nelement())
#     )
# )
# print(
#     "Sparsity in conv2.weight: {:.2f}%".format(
#         100. * float(torch.sum(model.feat.conv2.weight == 0))
#         / float(model.feat.conv2.weight.nelement())
#     )
# )
# print(
#     "Sparsity in fc1.weight: {:.2f}%".format(
#         100. * float(torch.sum(model.classifer.fc1.weight == 0))
#         / float(model.classifer.fc1.weight.nelement())
#     )
# )
# print(
#     "Sparsity in fc2.weight: {:.2f}%".format(
#         100. * float(torch.sum(model.classifer.fc2.weight == 0))
#         / float(model.classifer.fc2.weight.nelement())
#     )
# )
# print(
#     "Sparsity in fc3.weight: {:.2f}%".format(
#         100. * float(torch.sum(model.classifer.fc3.weight == 0))
#         / float(model.classifer.fc3.weight.nelement())
#     )
# )
# print(
#     "Global sparsity: {:.2f}%".format(
#         100. * float(
#             torch.sum(model.feat.conv1.weight == 0)
#             + torch.sum(model.feat.conv2.weight == 0)
#             + torch.sum(model.classifer.fc1.weight == 0)
#             + torch.sum(model.classifer.fc2.weight == 0)
#             + torch.sum(model.classifer.fc3.weight == 0)
#         )
#         / float(
#             model.feat.conv1.weight.nelement()
#             + model.feat.conv2.weight.nelement()
#             + model.classifer.fc1.weight.nelement()
#             + model.classifer.fc2.weight.nelement()
#             + model.classifer.fc3.weight.nelement()
#         )
#     )
# )



# # VGG19
# model = load_model("vgg19", "pytorch/vision:v0.16.0")

# parameters_to_prune = get_layers_for_pruning(model)

# parameters_to_prune = [(layer, "weight") for layer, _ in parameters_to_prune if isinstance(layer, torch.nn.modules.conv.Conv2d)\
#             or isinstance(layer, torch.nn.modules.pooling.MaxPool2d)\
#             or isinstance(layer, torch.nn.modules.linear.Linear)]

# prune.global_unstructured(
#     parameters_to_prune,
#     pruning_method=prune.L1Unstructured,
#     amount=0.5,
# )

# 
# YOLOV5 MODEL DETECTION
from ultralytics import YOLO
# model = torch.hub.load('ultralytics/yolov5', 'yolov5su', pretrained=True)

model = YOLO("yolov5su.pt")
parameters_to_prune = get_layers_for_pruning(model)
parameters_to_prune = [(layer, "weight") for layer, _ in parameters_to_prune if isinstance(layer, torch.nn.modules.conv.Conv2d) or isinstance(layer, torch.nn.modules.linear.Linear)]
# print(parameters_to_prune)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)
model.save("pruned_30_yolov5su.pt")

# print(list(parameters_to_prune[0][0].named_buffers()))
# model = torch.load("pruned_yolov5su.pt")  # local model
model = YOLO("pruned_30_yolov5su.pt")
# Run evaluation on COCO dataset
results = model.val(data="coco.yaml")


# model = torch.load("yolov5su.pt")  # local model, yolo weight
# parameters_to_prune = get_layers_for_pruning(model['model'])
# parameters_to_prune = [(layer, "weight") for layer, _ in parameters_to_prune if isinstance(layer, torch.nn.modules.conv.Conv2d) or isinstance(layer, torch.nn.modules.linear.Linear)]

# prune.global_unstructured(
#     parameters_to_prune,
#     pruning_method=prune.L1Unstructured,
#     amount=0.3,
# )
# model.save("pruned_yolov5su.pt")