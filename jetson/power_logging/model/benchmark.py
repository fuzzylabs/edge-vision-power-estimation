"""
Benchmark PyTorch models.

Script uses PyTorch to benchmark models and will support CUDA if it is available on the system
"""

import argparse
import json
import shutil
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch_tensorrt
from pydantic import BaseModel
from tqdm import tqdm

from model.model_utils import get_layers, load_model
from model.trt_utils import CustomProfiler, save_engine_info, save_layer_wise_profiling

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


class DetectionBenchmarkMetrics(BaseModel):
    config: dict[str, Any]
    total_time: float  # in seconds
    timestamp: str
    start_time: float
    end_time: float


class ClassifyBenchmarkMetrics(BaseModel):
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
        layer.register_forward_pre_hook(
            partial(layer_time_pre_hook, layer_time_dict, layer_name, start_event)
        )
        layer.register_forward_hook(
            partial(
                layer_time_hook, layer_time_dict, layer_name, start_event, end_event
            )
        )

    return layer_time_dict


def layer_time_pre_hook(
    layer_time_dict, layer_name, start_event: CudaEvent, module, input
) -> None:
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


def layer_time_hook(
    layer_time_dict, layer_name, start_event, end_event, module, input, output
) -> None:
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


def benchmark_detection(args: argparse.Namespace) -> None:
    """Benchmark latency and throughput for object detection models.

    Args:
        args: Arguments from CLI.
    """
    print("Starting benchmark...")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    try:
        model = load_model(args.model)

        print("Starting timing inference ...")
        start_event = CudaEvent(enable_timing=True)
        end_event = CudaEvent(enable_timing=True)

        save_dir = Path(args.result_dir) / args.model
        save_dir.mkdir(exist_ok=True, parents=True)

        # Clear ultralytics output if it exists
        if (save_dir / "val").exists():
            shutil.rmtree(save_dir / "val")

        ## Used for ONNX cpu time
        # start = time.time()
        # s = time.perf_counter()
        start_event.record()
        validation_results = model.val(
            data=args.dataset_name,
            project=save_dir,
            # device="cpu" used only for ONNX models
        )
        end_event.record()
        ## Used for ONNX cpu time
        # end = time.time()
        # total_time = time.perf_counter() - s

        if IS_GPU:
            torch.cuda.synchronize()

        print("Benchmarking complete ...")
        total_time = start_event.elapsed_time(end_event)

        results = DetectionBenchmarkMetrics(
            config=vars(args),
            total_time=total_time,  # in seconds
            timestamp=timestamp,
            start_time=start_event.get_time_stamp(),
            end_time=end_event.get_time_stamp(),
        )

        model_dir = f"{args.result_dir}/{args.model}"
        Path(model_dir).mkdir(exist_ok=True, parents=True)
        file_name = f"{args.model}.json"
        file_path = f"{model_dir}/{file_name}"
        with open(file_path, "w", encoding="utf-8") as outfile:
            json.dump(results.model_dump(), outfile, indent=4)

        validation_dict = {
            "metrics": validation_results.results_dict,
            "speed": validation_results.speed,
        }

        with open(
            f"{model_dir}/validation_results.json", "w"
        ) as validation_results_file:
            json.dump(validation_dict, validation_results_file, indent=4)

    except Exception as e:
        print(f"An error has occurred during benchmarking: {e}")
        raise e


def benchmark_classify(args: argparse.Namespace) -> None:
    """Benchmark latency and throughput across all backends.

    Args:
        args: Arguments from CLI.
    """
    print("Starting benchmark...")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    try:
        input_data = torch.randn(args.input_shape, device=DEVICE)
        model = load_model(args.model)
        model.eval().to(DEVICE)

        if args.dtype == "float16":
            dtype = torch.float16
        if args.dtype == "bfloat16":
            dtype = torch.bfloat16
        if args.dtype == "float32":
            dtype = torch.float32

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
        print(f"Warm complete in {time.perf_counter() - st:.2f} sec ...")

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

        results = ClassifyBenchmarkMetrics(
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
        with open(
            f"{model_dir}/{args.model}_layerwise_latency.json", "w"
        ) as layer_profiles_file:
            json.dump(layer_profiles, layer_profiles_file)
    except Exception as e:
        print(f"An error has occurred during benchmarking: {e}")
        return


def benchmark_trt(args: argparse.Namespace) -> None:
    """Benchmark latency and throughput across all backends.

    Additionally for tensorrt backend, we calculate layer-wise
    latency.

    Args:
        args: Arguments from CLI.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    input_data = torch.randn(args.input_shape, device=DEVICE)
    model = load_model(args.model)
    model.eval().to(DEVICE)

    if args.dtype == "float16":
        dtype = torch.float16
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    if args.dtype == "float32":
        dtype = torch.float32

    input_data = input_data.to(dtype)
    model = model.to(dtype)
    print(f"Using {DEVICE=} for benchmarking")

    exp_program = torch.export.export(model, tuple([input_data]))
    model = torch_tensorrt.dynamo.compile(
        exported_program=exp_program,
        inputs=[input_data],
        min_block_size=args.min_block_size,
        optimization_level=args.optimization_level,
        enabled_precisions={dtype},
        # Set to True for verbose output
        # NOTE: Performance Regression when rich library is available
        # https://github.com/pytorch/TensorRT/issues/3215
        debug=True,
        # Setting it to True returns PythonTorchTensorRTModule which has different profiling approach
        use_python_runtime=True,
    )

    print("Sleeping for 5 seconds to cool down...")
    time.sleep(5)

    st = time.perf_counter()
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(input_data)
    print(f"Warm complete in {time.perf_counter() - st:.2f} sec ...")

    print("Start timing using tensorrt backend ...")
    torch.cuda.synchronize()
    # Recorded in milliseconds
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.runs)]

    with torch.no_grad():
        for i in tqdm(range(args.runs)):
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

    save_layer_wise_profiling(mod, profiling_dir)
    save_engine_info(mod, profiling_dir)

    # Convert milliseconds to seconds
    timings = [s.elapsed_time(e) * 1.0e-3 for s, e in zip(start_events, end_events)]
    avg_throughput = args.input_shape[0] / np.mean(timings)
    print("Benchmarking complete ...")
    # Convert milliseconds to seconds
    total_exp_time = start.elapsed_time(end) * 1.0e-3
    print(f"Total time for experiment: {total_exp_time} sec")

    results = ClassifyBenchmarkMetrics(
        config=vars(args),
        total_time=total_exp_time,  # in seconds
        timestamp=timestamp,
        latencies=timings,  # in seconds
        avg_throughput=avg_throughput,
        avg_latency=np.mean(timings),  # in seconds
    )

    model_dir = f"{args.result_dir}/{args.model}"
    Path(model_dir).mkdir(exist_ok=True, parents=True)
    file_name = f"{args.model}_tensorrt.json"
    file_path = f"{model_dir}/{file_name}"
    with open(file_path, "w", encoding="utf-8") as outfile:
        json.dump(results.model_dump(), outfile, indent=4)
