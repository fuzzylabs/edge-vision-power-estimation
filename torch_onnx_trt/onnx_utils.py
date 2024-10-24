"""Benchmarking onnx model related utility functions."""

from pathlib import Path
import time
from tqdm import tqdm
from typing import Any
import torch
import numpy as np
from pydantic import BaseModel
import onnxruntime as ort

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_MAPPING = {"float32": "fp32", "float16": "fp16", "bfloat16": "bfp16"}


class BenchmarkMetrics(BaseModel):
    config: dict[str, Any] = {}
    total_time: float = 0.0  # in seconds
    timestamp: str = ""
    onnx_convert_time: float
    latencies: list[float]  # in seconds
    avg_latency: float  # in seconds
    avg_throughput: float


def export_to_onnx(
    input_data,
    pt_model,
    onnx_file_path: str,
) -> None:
    """Export PyTorch model to ONNX model

    Args:
        input_data: Input data used for tracing
        pt_model: PyTorch model
        onnx_file_path: Path to save ONNX model
    """
    torch.onnx.export(
        pt_model,  # model being run
        input_data,  # model input (or a tuple for multiple inputs)
        onnx_file_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },  # variable length axes
    )


def time_and_create_session(args, model_path):
    st = time.perf_counter()
    # Run ONNX inference on GPU
    providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": torch.cuda.current_device(),
                "user_compute_stream": str(torch.cuda.current_stream().cuda_stream),
            },
        )
    ]

    # Enable layer-wise profiling for ONNX models
    session_options = ort.SessionOptions()
    profiling_dir = f"{args.result_dir}/{args.model}/{args.backend}_profiling"
    Path(profiling_dir).mkdir(exist_ok=True, parents=True)
    # ONNX profiling in nanosecond
    session_options.enable_profiling = True
    session_options.profile_file_prefix = f"{profiling_dir}/{args.backend}_profiling"

    session = ort.InferenceSession(
        model_path,
        providers=providers,
        sess_options=session_options,
    )
    return session, time.perf_counter() - st


def benchmark_onnx(args, model, input_data) -> BenchmarkMetrics:
    st = time.perf_counter()
    dtype = DTYPE_MAPPING.get(args.dtype)
    onnx_dir_path = f"{args.model_dir}/{args.model}"
    onnx_file_path = f"{onnx_dir_path}/{args.model}_{dtype}.onnx"
    Path(onnx_dir_path).mkdir(exist_ok=True, parents=True)
    export_to_onnx(pt_model=model, input_data=input_data, onnx_file_path=onnx_file_path)
    onnx_convert_time = time.perf_counter() - st
    print(f"Time to taken convert PyTorch model to ONNX {onnx_convert_time:.3f} sec")

    st = time.perf_counter()
    ort_session, onnx_load_time = time_and_create_session(args, onnx_file_path)
    print(f"Time to taken load ONNX model from disk {onnx_load_time:.3f} sec")

    # Run warmup
    print(f"Using {DEVICE=} for benchmarking")
    st = time.perf_counter()
    ort_inputs = {ort_session.get_inputs()[0].name: input_data.cpu().numpy()}
    print("Warm up ...")
    for _ in range(args.warmup):
        _ = ort_session.run(None, ort_inputs)
    print(f"Warm complete in {time.perf_counter()-st:.2f} sec ...")

    # Run inference
    print(f"Start timing using backend {args.backend} ...")
    torch.cuda.synchronize()
    # Recorded in milliseconds
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.runs)]
    for i in tqdm(range(args.runs)):
        start_events[i].record()
        _ = ort_session.run(None, ort_inputs)
        end_events[i].record()
    torch.cuda.synchronize()
    # ONNX profiling in nanosecond
    ort_session.end_profiling()

    # Convert milliseconds to seconds
    timings = [s.elapsed_time(e) * 1.0e-3 for s, e in zip(start_events, end_events)]
    avg_throughput = args.input_shape[0] / np.mean(timings)
    results = BenchmarkMetrics(
        onnx_convert_time=onnx_convert_time,  # in seconds
        latencies=timings,  # in seconds
        avg_throughput=avg_throughput,
        avg_latency=np.mean(timings),  # in seconds
    )
    return results
