"""Experiment on model pruning using pytorch."""

import torch
import torch.nn.utils.prune as prune

from model.model_utils import load_model


model = load_model("lenet", "pytorch/vision:v0.16.0")

parameters_to_prune = (
    (model.feat.conv1, 'weight'),
    (model.feat.conv2, 'weight'),
    (model.classifer.fc1, 'weight'),
    (model.classifer.fc2, 'weight'),
    (model.classifer.fc3, 'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)

print(
    "Sparsity in conv1.weight: {:.2f}%".format(
        100. * float(torch.sum(model.feat.conv1.weight == 0))
        / float(model.feat.conv1.weight.nelement())
    )
)
print(
    "Sparsity in conv2.weight: {:.2f}%".format(
        100. * float(torch.sum(model.feat.conv2.weight == 0))
        / float(model.feat.conv2.weight.nelement())
    )
)
print(
    "Sparsity in fc1.weight: {:.2f}%".format(
        100. * float(torch.sum(model.classifer.fc1.weight == 0))
        / float(model.classifer.fc1.weight.nelement())
    )
)
print(
    "Sparsity in fc2.weight: {:.2f}%".format(
        100. * float(torch.sum(model.classifer.fc2.weight == 0))
        / float(model.classifer.fc2.weight.nelement())
    )
)
print(
    "Sparsity in fc3.weight: {:.2f}%".format(
        100. * float(torch.sum(model.classifer.fc3.weight == 0))
        / float(model.classifer.fc3.weight.nelement())
    )
)
print(
    "Global sparsity: {:.2f}%".format(
        100. * float(
            torch.sum(model.feat.conv1.weight == 0)
            + torch.sum(model.feat.conv2.weight == 0)
            + torch.sum(model.classifer.fc1.weight == 0)
            + torch.sum(model.classifer.fc2.weight == 0)
            + torch.sum(model.classifer.fc3.weight == 0)
        )
        / float(
            model.feat.conv1.weight.nelement()
            + model.feat.conv2.weight.nelement()
            + model.classifer.fc1.weight.nelement()
            + model.classifer.fc2.weight.nelement()
            + model.classifer.fc3.weight.nelement()
        )
    )
)