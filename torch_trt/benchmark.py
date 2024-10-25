"""Script to collect latency, throughput and layer-wise latency for CNN models.

To run benchmark script:
    python benchmark.py \
        --backend <backend> \
        --model <pytorch_hub_model_name> \
        --dtype <dtype> \
        --save-result

"""

from rich import print
from tqdm import tqdm
from typing import Any
from datetime import datetime
import torch
import time
from pathlib import Path
import torch.backends.cudnn as cudnn
import torch_tensorrt
import numpy as np
import argparse
import json
from pydantic import BaseModel
from trt_utils import CustomProfiler, save_engine_info, save_layer_wise_profiling

cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKENDS = [
    "pytorch",
    "torchscript",
    "tensorrt",
]


class BenchmarkMetrics(BaseModel):
    config: dict[str, Any]
    total_time: float  # in seconds
    timestamp: str
    latencies: list[float]  # in seconds
    avg_latency: float  # in seconds
    avg_throughput: float


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
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    input_data = torch.randn(args.input_shape, device=DEVICE)
    model = load_model(args.model)
    model.eval().to(DEVICE)

    dtype = torch.float32
    if args.dtype == "float16":
        dtype = torch.float16
    if args.dtype == "bfloat16":
        dtype = torch.float16

    input_data = input_data.to(dtype)
    model = model.to(dtype)

    if args.backend == "torchscript":
        model = torch.jit.trace(model, input_data)

    if args.backend == "tensorrt":
        # Run the model on an input to cause compilation
        exp_program = torch.export.export(model, tuple([input_data]))
        model = torch_tensorrt.dynamo.compile(
            exported_program=exp_program,
            inputs=[input_data],
            min_block_size=args.min_block_size,
            optimization_level=args.optimization_level,
            enabled_precisions={dtype},
            # Set to True for verbose output
            debug=True,
            # Setting it to True returns PythonTorchTensorRTModule which has different profiling approach
            use_python_runtime=True,
        )

    print(f"Using {DEVICE=} for benchmarking")
    st = time.perf_counter()
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(input_data)
    print(f"Warm complete in {time.perf_counter()-st:.2f} sec ...")

    print(f"Start timing using backend {args.backend} ...")
    torch.cuda.synchronize()
    # Recorded in milliseconds
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.runs)]

    with torch.no_grad():
        for i in tqdm(range(args.runs)):

            if args.backend == "tensorrt":
                # Hack for enabling profiling
                # https://github.com/pytorch/TensorRT/issues/1467
                profiling_dir = f"{args.result_dir}/{args.model}/trt_profiling"
                Path(profiling_dir).mkdir(exist_ok=True, parents=True)

                # Records traces in milliseconds
                # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Profiler.html#tensorrt.Profiler
                mod = list(model.named_children())[0][1]
                mod.enable_profiling(profiler=CustomProfiler())

            start_events[i].record()
            _ = model(input_data)
            end_events[i].record()

        end.record()
        torch.cuda.synchronize()

    # Save layer-wise latency and engine information to json file.
    if args.backend == "tensorrt":
        save_layer_wise_profiling(mod, profiling_dir)
        save_engine_info(mod, profiling_dir)

    # Convert milliseconds to seconds
    timings = [s.elapsed_time(e) * 1.0e-3 for s, e in zip(start_events, end_events)]
    avg_throughput = args.input_shape[0] / np.mean(timings)
    print("Benchmarking complete ...")
    # Convert milliseconds to seconds
    total_exp_time = start.elapsed_time(end) * 1.0e-3
    print(f"Total time for experiment: {total_exp_time} sec")

    results = BenchmarkMetrics(
        config=vars(args),
        total_time=total_exp_time,  # in seconds
        timestamp=current_dt,
        latencies=timings,  # in seconds
        avg_throughput=avg_throughput,
        avg_latency=np.mean(timings),  # in seconds
    )

    if args.save_result:
        save_dir = f"{args.result_dir}/{args.model}"
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        file_name = f"{args.model}_{args.backend}_{current_dt}.json"
        file_path = f"{save_dir}/{file_name}"
        with open(file_path, "w", encoding="utf-8") as outfile:
            json.dump(results.model_dump_json(indent=4), outfile)

    print(results)
    if args.backend == "tensorrt":
        print(
            f"View the results of trace for '{profiling_dir}/_run_on_acc_0_engine_engine_exectuion_profile.trace' "
            "in UI at https://ui.perfetto.dev "
        )


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
        default="pytorch",
        choices=BACKENDS,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
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
        "--optimization-level",
        type=int,
        default=5,
        help="Builder optimization 0-5, higher levels imply longer build time, "
        "searching for more optimization options.",
    )
    parser.add_argument(
        "--min-block-size",
        type=int,
        default=5,
        help="Minimum number of operators per TRT-Engine Block",
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

    args = parser.parse_args()
    benchmark(args)
