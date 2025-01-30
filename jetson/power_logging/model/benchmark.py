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

import torch
from pydantic import BaseModel
from tqdm import tqdm

from functools import partial
from model.model_utils import load_model, get_layers
from model.zero_keep_pruning import zero_keep_pruning

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

    def __init__(self, enable_timing = True):
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
    timestamp: str
    latencies: list[float]  # in seconds
    avg_latency: float  # in seconds
    avg_throughput: float

def define_and_register_hooks(model, device) -> dict:
    """
        Define and register hooks with CUDA or CPU timing.

    Args:
        model: model we are registering hooks to its layers.
        device: CPU or GPU.

    Returns:
        Dictionary containing the result.
    """
    layer_time_dict = {}

    for layer_name, layer in get_layers(model):
        start_event = CudaEvent(enable_timing=True)
        end_event = CudaEvent(enable_timing=True)
        layer.register_forward_pre_hook(partial(layer_time_pre_hook, layer_time_dict, layer_name, start_event))
        layer.register_forward_hook(partial(layer_time_hook, layer_time_dict, layer_name, start_event, end_event))
    
    return layer_time_dict


def layer_time_pre_hook(layer_time_dict, layer_name, start_event: CudaEvent, module, input) -> None:
    """
    Pre-hook to record start time.

    Args:
        layer_time_dict: dictionary to save hook function output.
        layer_name: the layer to register hook.
        start_event: an instance of the CudaEvent object use for marking the start time of before an layer execution.
        module: the module to register hook.
        input: tuple containing the input arguments to module's forward method.
    """
    layer_time_dict[layer_name] = {}
    start_event.record()


def layer_time_hook(layer_time_dict, layer_name, start_event, end_event, module, input, output) -> None:
    """
    Hook to record end time and calculate duration.

    Args:
        layer_time_dict: dictionary to save hook function output.
        layer_name: the layer to register hook.
        start_event: the same instance of the CudaEvent object used in pre hook.
        end_event: start_event: an instance of the CudaEvent object use for marking the end time of a layer execution.
        module: the module to register hook.
        input: tuple containing the input arguments to module's forward method.
        output: the output tensor from the forward method.
    """
    end_event.record()
    if IS_GPU:
        torch.cuda.synchronize()
    elapsed = start_event.elapsed_time(end_event)
    layer_time_dict[layer_name]["elapsed_time"] = elapsed
    layer_time_dict[layer_name]["start_time"] = start_event.get_time_stamp()


def benchmark(args: argparse.Namespace) -> None:
    """Benchmark latency and throughput across all backends.

    Args:
        args: Arguments from CLI.
    """
    print("Starting benchmark with ZKFP...")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    try:
        input_data = torch.randn(args.input_shape, device=DEVICE)
        model = load_model(args.model, args.model_repo)
        model.eval().to(DEVICE)

        if args.use_zkp:
            model, pruning_masks = zero_keep_pruning(model, threshold=0.0)
            print("Pruning applied")
        else:
            print("Running baseline")


        dtype = torch.float32
        if args.dtype == "float16":
            dtype = torch.float16
        if args.dtype == "bfloat16":
            dtype = torch.bfloat16

        input_data = input_data.to(dtype)
        model = model.to(dtype)
        print(f"Using {DEVICE=} for benchmarking")
        if DEVICE == "cpu":
            print("Warning: Running on CPU.")

        st = time.perf_counter()
        print("Warm up ...")
        with torch.no_grad():
            for _ in range(args.warmup):
                _ = model(input_data)
        print(f"Warm complete in {time.perf_counter()-st:.2f} sec ...")

        layer_profiles = []
        layer_profile = define_and_register_hooks(model, DEVICE)

        print("Starting timing inference ...")
        latencies = []
        start_events = [CudaEvent(enable_timing=True) for _ in range(args.runs)]
        end_events = [CudaEvent(enable_timing=True) for _ in range(args.runs)]
        
        with torch.no_grad():
            for i in tqdm(range(args.runs)):
                start_events[i].record()
                _ = model(input_data)
                end_events[i].record()

                if IS_GPU:
                    torch.cuda.synchronize()

                latency = start_events[i].elapsed_time(end_events[i])
                latencies.append(latency * 1.0e-3)
                layer_profiles.append(layer_profile.copy())

        print("Benchmarking complete ...")

        total_time = sum(latencies)
        avg_latency = total_time / len(latencies)
        avg_throughput = args.input_shape[0] / avg_latency


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

        if args.use_zkp:
            output_filename = f"{args.model}_zkp_results.json"
        else:
            output_filename = f"{args.model}_baseline_results.json"

        output_path = f"{model_dir}/{output_filename}"

        with open(output_path, "w", encoding="utf-8") as outfile:  
            json.dump(results.dict(), outfile, indent=4)

        with open(f"{model_dir}/{args.model}_layerwise_latency.json", "w") as layer_profiles_file:
            json.dump(layer_profiles, layer_profiles_file)
        
        print("Results saved")
    except Exception as e:
        print(f"An error has occurred during benchmarking: {e}")
        return
