"""Script to collect latency, throughput and layer-wise latency for CNN models.

To run benchmark script:
    python benchmark.py \
        --backend <backend> \
        --model <pytorch_hub_model_name> \
        --dtype <dtype> \
        --save-result

"""

import json
from rich import print
from pathlib import Path
from typing import Any
from datetime import datetime
import time
import torch
import torchvision
import torch.backends.cudnn as cudnn
import argparse
from onnx_utils import benchmark_onnx
from trt_utils import benchmark_trt

cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKENDS = [
    "onnx",
    "tensorrt",
]


def load_model(model_name: str) -> Any:
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


def benchmark(args: argparse.Namespace) -> None:
    """Benchmark latency and throughput across all backends.

    Additionally for tensorrt backend, we calculate layer-wise
    latency.

    Args:
        args: Arguments from CLI.
    """
    start_exp = time.perf_counter()

    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    input_data = torch.randn(args.input_shape, device=DEVICE)
    model = load_model(args.model)
    model.eval().to(DEVICE)

    dtype = torch.float32
    if args.dtype == "float16":
        dtype = torch.float16
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16

    input_data = input_data.to(dtype)
    model = model.to(dtype)

    if args.backend == "onnx":
        # Convert PyTorch to ONNX and benchmark
        results = benchmark_onnx(args, model, input_data)
    if args.backend == "tensorrt":
        results = benchmark_trt(args, input_data)

    end_exp = time.perf_counter()
    print("Benchmarking complete ...")
    # Convert milliseconds to seconds
    total_exp_time = end_exp - start_exp
    print(f"Total time for experiment: {total_exp_time} sec")

    results.config = vars(args)
    results.timestamp = current_dt
    results.total_time = total_exp_time

    if args.save_result:
        save_dir = f"{args.result_dir}/{args.model}"
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        file_name = f"{args.model}_{args.backend}_{current_dt}.json"
        file_path = f"{save_dir}/{file_name}"
        with open(file_path, "w", encoding="utf-8") as outfile:
            json.dump(results.model_dump(), outfile, indent=4)

    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Benchmarking CNN models",
        description="Collect latency, throughput and layer-wise time for imagenet pretrained CNN models",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="alexnet",
        help="Specify name of pretrained CNN model from PyTorch Hub."
        "For more information on PyTorch Hub visit: "
        "https://pytorch.org/hub/research-models",
    )
    parser.add_argument(
        "--backend",
        type=str,
        help="Backend to use for benchmarking",
        default="onnx",
        choices=BACKENDS,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32", "int8"],
        help="Data type for model weights and activations.\n\n"
        '* "float16" is the same as "half".\n'
        '* "bfloat16" for a balance between precision and range.\n'
        '* "float32" for FP32 precision.',
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs="+",
        default=[1, 3, 224, 224],
        help="Input shape BCHW",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of iterations to perform warmup before benchmarking",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of iterations to benchmarking "
        "to collect latency, throughput and layer-wise latency",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="results",
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Specify directory to save ONNX models."
        "If not specified, models are saved in the current directory.",
    )

    args = parser.parse_args()
    benchmark(args)
