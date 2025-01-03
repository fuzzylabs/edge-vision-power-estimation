import argparse

from model.benchmark import load_model
from ablation.ablate import get_layers_for_ablation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect layers for ablation.')

    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="Specify name of pretrained CNN model from PyTorch Hub."
        "For more information on PyTorch Hub visit: "
        "https://pytorch.org/hub/research-models",
    )

    args = parser.parse_args()

    model = load_model(args.model)
    layers = get_layers_for_ablation(model)
    for layer in layers:
        print(",".join(layer))