"""
Benchmark PyTorch models.

Script uses PyTorch to benchmark models and will support CUDA if it is available on the system
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pydantic import BaseModel
from tqdm import tqdm

from model.lenet import LeNet

"""
Wrapper class for Torch.cuda.event for non-CUDA supported devices

Methods:
    - record(): Records an event if CUDA is available
    - elapsed_time(): Calculates elapsed time between events
    - synchronize(): synchronizes events in instance of CUDA
"""
class CudaEvent:
    def __init__(self, enable_timing = True):
        if torch.cuda.is_available():
            self.event = torch.cuda.Event(enable_timing=enable_timing)
        else:
            print("Warning: CUDA not available. Instance outimed.")
            self.event = None 

    def record(self):
        if self.event:
            self.event.record()

    def elapsed_time(self, n_event):
        if self.event and n_event.event:
            return self.event.elapsed_time(n_event.event)
        return 0
    
    def synchronize(self):
        if self.event:
            self.event.synchronize()


cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BenchmarkMetrics(BaseModel):
    config: dict[str, Any]
    total_time: float  # in seconds
    timestamp: str
    latencies: list[float]  # in seconds
    avg_latency: float  # in seconds
    avg_throughput: float


def load_model(model_name: str, model_repo: str) -> Any:
    """Load model from Pytorch Hub.

    Args:
        model_name: Name of model.
            It should be same as that in Pytorch Hub.

    Raises:
        ValueError: If loading model fails from PyTorch Hub

    Returns:
        PyTorch model
    """
    if model_name == "lenet":
        return LeNet()
    if model_name == "fcn_resnet50":
        return torch.hub.load(model_repo, model_name, pretrained=True)
    try:
        return torch.hub.load(model_repo, model_name)
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
    start = CudaEvent(enable_timing=True)
    end = CudaEvent(enable_timing=True)
    start.record()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    input_data = torch.randn(args.input_shape, device=DEVICE)
    model = load_model(args.model, args.model_repo)
    model.eval().to(DEVICE)

    dtype = torch.float32
    if args.dtype == "float16":
        dtype = torch.float16
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16

    input_data = input_data.to(dtype)
    model = model.to(dtype)
    print(f"Using {DEVICE=} for benchmarking")

    st = time.perf_counter()
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(input_data)
    print(f"Warm complete in {time.perf_counter()-st:.2f} sec ...")

    print("Start timing using tensorrt backend ...")
    
    with torch.no_grad():
        for i in tqdm(range(args.runs)):
            # Hack for enabling profiling
            # https://github.com/pytorch/TensorRT/issues/1467
            profiling_dir = f"{args.result_dir}/{args.model}/trt_profiling"
            Path(profiling_dir).mkdir(exist_ok=True, parents=True)
            _ = model(input_data)

    print("Benchmarking complete ...")


    results = BenchmarkMetrics(
        config=vars(args),
        total_time=0, #total_exp_time,  # in seconds
        timestamp=timestamp,
        latencies=[], #timings,  # in seconds
        avg_throughput=0, #avg_throughput,
        avg_latency=0, #np.mean(timings),  # in seconds
    )

    model_dir = f"{args.result_dir}/{args.model}"
    Path(model_dir).mkdir(exist_ok=True, parents=True)
    file_name = f"{args.model}_tensorrt.json"
    file_path = f"{model_dir}/{file_name}"
    with open(file_path, "w", encoding="utf-8") as outfile:
        json.dump(results.model_dump(), outfile, indent=4)
