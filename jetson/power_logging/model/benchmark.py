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
            print("Warning: CUDA not available.")
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

    Args:
        args: Arguments from CLI.
    """
    print("Starting the benchmarking process...")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    try:
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
        if DEVICE == "cpu":
            print("WARNING: Running on CPU. Timing may vary")

        print("Warm up ...")
        st = time.perf_counter()
        with torch.no_grad():
            for _ in range(args.warmup):
                _ = model(input_data)
        print(f"Warm complete in {time.perf_counter()-st:.2f} sec ...")

        print("Start timing using pytorch backend ...")
        latencies = []
        start_events = [CudaEvent(enable_timing=True) for _ in range(args.runs)]
        end_events = [CudaEvent(enable_timing=True) for _ in range(args.runs)]

        with torch.no_grad():
            for i in tqdm(range(args.runs)):
                start_events[i].record()
                _ = model(input_data)
                end_events[i].record()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                latency = start_events[i].elapsed_time(end_events[i])
                latencies.append(latency * 1.0e-3)

        print("Benchmarking complete ...")

        total_time = sum(latencies) # Total time for all executions
        avg_latency = total_time / len(latencies) # Average latency per execution
        avg_throughput = args.input_shape[0] / avg_latency # Throughput in samples/sec


        results = BenchmarkMetrics(
            config=vars(args),
            total_time=total_time,  # in seconds
            timestamp=timestamp,
            latencies=latencies,  # in seconds
            avg_throughput=avg_throughput,
            avg_latency=avg_latency,  # in seconds
        )

        model_dir = f"{args.result_dir}/{args.model}"
        Path(model_dir).mkdir(exist_ok=True, parents=True)
        file_name = f"{args.model}_pytorch.json"
        file_path = f"{model_dir}/{file_name}"
        with open(file_path, "w", encoding="utf-8") as outfile:
            json.dump(results.model_dump(), outfile, indent=4)
    except Exception as e:
        print(f"An error occured during benchmarking: {e}")
        return
