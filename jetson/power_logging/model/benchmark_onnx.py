"""
Benchmark ONNX models.

Script uses ONNX to benchmark models and will support CUDA if it is available on the system
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from pydantic import BaseModel
from tqdm import tqdm

"""
Wrapper class for Torch.cuda.event for non-CUDA supported devices

Methods:
    - record(): Records an event if CUDA is available
    - elapsed_time(): Calculates elapsed time between events
"""


class CudaEvent:
    start_time: float
    time_stamp: float
    event: torch.cuda.Event | None

    def __init__(self, enable_timing=True):
        if IS_GPU:
            self.event = torch.cuda.Event(enable_timing=enable_timing)
        else:
            print("Warning: CUDA not available.")
            self.event = None

    def record(self):
        self.start_time = time.time()
        self.time_stamp = time.perf_counter()

        if self.event:
            self.event.record()

    def elapsed_time(self, n_event):
        if self.event and n_event.event:
            return self.event.elapsed_time(n_event.event)
        else:
            return n_event.time_stamp - self.time_stamp

    def get_time_stamp(self):
        return self.start_time


IS_GPU = torch.cuda.is_available()
DEVICE = "cuda" if IS_GPU else "cpu"


class BenchmarkMetrics(BaseModel):
    config: dict[str, Any]
    total_time: float  # in seconds
    timestamps: tuple
    latencies: list[float]  # in seconds
    avg_latency: float  # in seconds
    avg_throughput: float


def benchmark(args: argparse.Namespace) -> None:
    """Benchmark latency and throughput across all backends.

    Args:
        args: Arguments from CLI.
    """
    print("Starting benchmark...")

    try:
        input_data = torch.randn(args.input_shape, device=DEVICE, dtype=torch.float32)
        session = ort.InferenceSession(
            args.model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        # Get the model inputs
        print(f"Using {DEVICE=} for benchmarking")
        if DEVICE == "cpu":
            print("Warning: Running on CPU.")

        model_outputs = session.get_outputs()
        model_inputs = session.get_inputs()
        io_binding = session.io_binding()
        st = time.perf_counter()
        print("Warm up ...")
        for _ in range(100):
            io_binding.bind_input(
                name=model_inputs[0].name,
                device_type="cuda",
                device_id=0,
                element_type=np.float32,
                shape=tuple(input_data.shape),
                buffer_ptr=input_data.data_ptr(),
            )
            io_binding.bind_output(name=model_outputs[0].name)
            session.run_with_iobinding(io_binding)

        print(f"Warm complete in {time.perf_counter() - st:.2f} sec ...")

        time.sleep(10)
        print("Sleeping for 10 seconds")

        model_profiles = []
        print("Starting timing inference ...")
        latencies = []
        start_events = [CudaEvent(enable_timing=True) for _ in range(args.runs)]
        end_events = [CudaEvent(enable_timing=True) for _ in range(args.runs)]

        for i in tqdm(range(300)):
            start_events[i].record()
            io_binding.bind_input(
                name=model_inputs[0].name,
                device_type="cuda",
                device_id=0,
                element_type=np.float32,
                shape=tuple(input_data.shape),
                buffer_ptr=input_data.data_ptr(),
            )
            io_binding.bind_output(name=model_outputs[0].name)
            session.run_with_iobinding(io_binding)
            end_events[i].record()

            if IS_GPU:
                torch.cuda.synchronize()

            model_profiles.append(
                (start_events[i].get_time_stamp(), end_events[i].get_time_stamp())
            )
            latency = start_events[i].elapsed_time(end_events[i])
            latencies.append(latency * 1.0e-3)
            time.sleep(1)

        print("Benchmarking complete ...")

        total_time = sum(latencies)
        avg_latency = total_time / len(latencies)
        avg_throughput = args.input_shape[0] / avg_latency

        results = BenchmarkMetrics(
            config=vars(args),
            total_time=total_time,  # in seconds
            timestamps=model_profiles,
            latencies=latencies,  # in seconds
            avg_throughput=avg_throughput,
            avg_latency=avg_latency,  # in seconds
        )

        model_dir = f"{args.result_dir}/{args.model}"
        Path(model_dir).mkdir(exist_ok=True, parents=True)
        file_name = f"{args.model}_onnx.json"
        file_path = f"{model_dir}/{file_name}"
        with open(file_path, "w", encoding="utf-8") as outfile:
            json.dump(results.model_dump(), outfile, indent=4)
    except Exception as e:
        print(f"An error has occurred during benchmarking: {e}")
        return
