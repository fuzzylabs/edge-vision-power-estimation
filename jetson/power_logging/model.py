"""Script to load a PyTorch CNN model and convert it to TensorRT model."""

from typing import Any
import torch
import torch_tensorrt
import argparse


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_from_torch_hub(model_name: str) -> Any:
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
        return torch.hub.load("pytorch/vision", model_name, weights="IMAGENET1K_V1")
    except:
        raise ValueError(
            f"Model name: {model_name} is most likely incorrect. "
            "Please refer https://pytorch.org/hub/ to get model name."
        )


def prepare_inference(args: argparse.Namespace):
    input_data = torch.randn(args.input_shape, device=DEVICE)
    model = load_model_from_torch_hub(args.model)
    model.eval().to(DEVICE)

    dtype = torch.float32
    if args.dtype == "float16":
        dtype = torch.float16
    if args.dtype == "bfloat16":
        dtype = torch.float16

    input_data = input_data.to(dtype)
    model = model.to(dtype)

    # Run the model on an input to cause compilation
    exp_program = torch.export.export(model, tuple([input_data]))
    model =  torch_tensorrt.dynamo.compile(
        exported_program=exp_program,
        inputs=[input_data],
        min_block_size=args.min_block_size,
        optimization_level=args.optimization_level,
        enabled_precisions={dtype},
        # Set to True for verbose output
        debug=True,
        # Setting it to True returns PythonTorchTensorRTModule which has different profiling approach
        use_python_runtime=False,
    )

    return input_data, model


def warm_up(warm_up_iterations: int, model, input_data):
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(warm_up_iterations):
            _ = model(input_data)
    print(f"Warm completed")