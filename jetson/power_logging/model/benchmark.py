"""
Benchmark PyTorch models.

Script uses PyTorch to benchmark models and will support CUDA if it is available on the system
"""

import argparse
import json
import time
import torch
import psutil

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from tqdm import tqdm

from functools import partial
from model.model_utils import load_model, get_layers
from model.zero_keep_pruning import zero_keep_pruning
from thop import profile
import shutil
from ultralytics import YOLO

if hasattr(profile, '_register_hooks'):
    del profile._register_hooks[:]

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
    warning_printed = False

    def __init__(self, enable_timing = True):
        if IS_GPU:
            self.event = torch.cuda.Event(enable_timing=enable_timing)
        else:
            if not CudaEvent.warning_printed:
                print("Warning: CUDA not available.")
                CudaEvent.warning_printed = True
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
    memory_usage: dict
    model_size: dict
    flops: float
    energy_efficiency: float
    start_time: float # new
    end_time: float # new 

def get_memory_usage():
    return {
        "cpu_memory": psutil.Process().memory_info().rss / (1024 ** 2),
        "gpu_memory": torch.cuda.memory_allocated() / (1024 ** 2) if IS_GPU else None
    }

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

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        input_data = torch.randn(args.input_shape, device=DEVICE)
        try:
            print(f"Attempting to load YOLO model: {args.model}")
            model = load_model(args.model)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")

        model.eval().to(DEVICE)

        if args.use_zkp:
            model, pruning_masks = zero_keep_pruning(model, threshold=0.0)
            print("Starting benchmark with ZKFP...")
        else:
            print("Starting benchmark...")


        dtype = torch.float32
        if args.dtype == "float16":
            dtype = torch.float16
        if args.dtype == "bfloat16":
            dtype = torch.bfloat16

        input_data = input_data.to(dtype)
        model = model.to(dtype)

        print("Starting...")
        if not hasattr(profile, '_hooks_registered'):
            profile._hooks_registered = True
            macs, params = profile(model, inputs=(input_data,))
            print("Profiling Finished...")
            total_flops = macs * 2
        else:
            print("Hooks already in use. Skipping")

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
        start_event = CudaEvent(enable_timing=True)
        end_event = CudaEvent(enable_timing=True)

        save_dir = Path(args.result_dir) / f"{args.model}_zkp" if args.use_zkp else Path(args.result_dir) / f"{args.model}_baseline"
        save_dir.mkdir(exist_ok=True, parents=True)

        validation_results = None
        try:
            start_event.record()
            validation_results = model.val(
                data=args.dataset_name,
                project=save_dir,
            )
            end_event.record()
        except Exception as val_error:
            print(f"Validation Failed: {val_error}")
        
        # Clear ultralytics output if it exists
        if (save_dir / "val").exists():
            shutil.rmtree(save_dir / "val")

        if IS_GPU:
            torch.cuda.synchronize()

        print("Benchmarking complete ...")


        latencies = [time.perf_counter() - time.perf_counter() for _ in range(args.runs)]
        total_time = sum(latencies)
        avg_latency = total_time / len(latencies)
        avg_throughput = args.input_shape[0] / avg_latency
        memory_usage = get_memory_usage()
        power_usage = args.power_usage if hasattr(args, 'power_usage') else 0
        energy_efficiency = power_usage / avg_throughput if power_usage else 0.0

        torch.save(model.state_dict(), "temp_model.pth")
        model_size = {
            "size_MB": Path("temp_model.pth").stat().st_size / (1024 ** 2), 
        }
        Path("temp_model.pth").unlink()

        output_path = save_dir / f"{args.model}_zkp_results.json" if args.use_zkp else save_dir / f"{args.model}_baseline_results.json"

        results = BenchmarkMetrics(
            config=vars(args),
            total_time=total_time,  # in seconds
            timestamp=timestamp,
            latencies=latencies,  # in seconds
            avg_latency=avg_latency,  # in seconds
            avg_throughput=avg_throughput,
            memory_usage=memory_usage,
            model_size=model_size,
            flops=total_flops,
            energy_efficiency=energy_efficiency,
            start_time=start_event.get_time_stamp(),
            end_time=end_event.get_time_stamp(),
            validation_results=validation_results,
        )


        # save_dir = Path(args.result_dir) / f"{args.model}_zkp" if args.use_zkp else Path(args.result_dir) / f"{args.model}_baseline"
        # save_dir.mkdir(exist_ok=True, parents=True)

        # model_dir = f"{args.result_dir}/{args.model}"
        # Path(model_dir).mkdir(exist_ok=True, parents=True)

        # output_path = f"{model_dir}/{args.model}_zkp_results.json" if args.use_zkp else f"{model_dir}/{args.model}_baseline_results.json"

        # if args.use_zkp:
        #     output_filename = f"{args.model}_zkp_results.json"
        # else:
        #     output_filename = f"{args.model}_baseline_results.json"

        # output_path = f"{model_dir}/{output_filename}"

        with open(output_path, "w", encoding="utf-8") as outfile:  
            json.dump(results.dict(), outfile, indent=4)

        # with open(f"{model_dir}/{args.model}_layerwise_latency.json", "w") as layer_profiles_file:
        #     json.dump(layer_profiles, layer_profiles_file)
        
        print("Benchmarking complete. Results saved at: ", output_path)
    except Exception as e:
        print(f"An error has occurred during benchmarking: {e}")
        return
