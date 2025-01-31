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
from pytorch.pruning import global_unstructured_prune
import shutil

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
    start_time: float
    end_time: float

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
    print("Starting benchmark...")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    try:
        model = load_model(args.model).to(DEVICE)

        if args.prune:
            print(f"Pruning model weight by {args.pruning_sparsity}")
            global_unstructured_prune(
                model=model,
                amount=args.pruning_sparsity
            )

        print("Starting timing inference ...")
        start_event = CudaEvent(enable_timing=True)
        end_event = CudaEvent(enable_timing=True)

        save_dir = Path(args.result_dir) / args.model
        save_dir.mkdir(exist_ok=True, parents=True)

        # Clear ultralytics output if it exists
        if (save_dir / "val").exists():
            shutil.rmtree(save_dir / "val")

        start_event.record()
        validation_results = model.val(
            data=args.dataset_name,
            project=save_dir,
        )
        end_event.record()

        if IS_GPU:
            torch.cuda.synchronize()

        print("Benchmarking complete ...")

        total_time = start_event.elapsed_time(end_event)

        results = BenchmarkMetrics(
            config=vars(args),
            total_time=total_time,  # in seconds
            timestamp=timestamp,
            start_time=start_event.get_time_stamp(),
            end_time=end_event.get_time_stamp(),
        )

        model_dir = f"{args.result_dir}/{args.model}"
        Path(model_dir).mkdir(exist_ok=True, parents=True)
        file_name = f"{args.model}_pytorch.json"
        file_path = f"{model_dir}/{file_name}"
        with open(file_path, "w", encoding="utf-8") as outfile:
            json.dump(results.model_dump(), outfile, indent=4)

        validation_dict = {
            "metrics": validation_results.results_dict,
            "speed": validation_results.speed,
        }

        with open(f"{model_dir}/validation_results.json", "w") as validation_results_file:
            json.dump(validation_dict, validation_results_file, indent=4)

    except Exception as e:
        raise e
        print(f"An error has occurred during benchmarking: {e}")
        return
