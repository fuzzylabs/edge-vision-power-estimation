"""Benchmarking trt model related utility functions."""

from pathlib import Path
import time
from tqdm import tqdm
from typing import Any
import torch
import numpy as np
from pydantic import BaseModel
from trt.infer import TensorRTInfer
from trt.runtime import cuda_call
from cuda import cudart

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_MAPPING = {"float32": "fp32", "float16": "fp16", "bfloat16": "bfp16"}


class BenchmarkMetrics(BaseModel):
    config: dict[str, Any] = {}
    total_time: float = 0.0  # in seconds
    timestamp: str = ""
    trt_load_time: float  # in seconds
    latencies: list[float]  # in seconds
    avg_latency: float  # in seconds
    avg_throughput: float


def benchmark_trt(args, input_data) -> BenchmarkMetrics:
    dtype = DTYPE_MAPPING.get(args.dtype)
    trt_engine_path = f"{args.model_dir}/{args.model}/{args.model}_{dtype}.engine"

    if not Path(trt_engine_path).exists():
        raise ValueError(
            f"TensoRT engine not found at {trt_engine_path}. "
            "Please run build_engine script to create a TensorRT engine first."
        )

    # Disable profiling for warmup
    st = time.perf_counter()
    trt_infer = TensorRTInfer(engine_path=trt_engine_path, enable_profiler=False)
    trt_load_time = time.perf_counter() - st
    print(f"Time to taken load TRT engine from disk {trt_load_time:.3f} sec")

    # Run warmup
    print(f"Using {DEVICE=} for benchmarking")
    st = time.perf_counter()
    print("Warm up ...")
    for _ in range(args.warmup):
        _ = trt_infer.infer([input_data.cpu().numpy()])
    print(f"Warm complete in {time.perf_counter()-st:.2f} sec ...")

    # Enable profiling recorded in milliseconds
    st = time.perf_counter()
    trt_infer = TensorRTInfer(engine_path=trt_engine_path, enable_profiler=True)
    trt_load_time = time.perf_counter() - st
    print(f"Time to taken load TRT engine from disk {trt_load_time:.3f} sec")

    # Run inference
    print(f"Start timing using backend {args.backend} ...")
    # Recorded in milliseconds
    start_events = [cuda_call(cudart.cudaEventCreate()) for _ in range(args.runs)]
    end_events = [cuda_call(cudart.cudaEventCreate()) for _ in range(args.runs)]
    for i in tqdm(range(args.runs)):
        cuda_call(cudart.cudaEventRecord(start_events[i], 0))
        _ = trt_infer.infer(input_data.cpu().numpy())
        cuda_call(cudart.cudaEventRecord(end_events[i], 0))
        cuda_call(cudart.cudaEventSynchronize(end_events[i]))

    # Save profiling information
    trt_profile_dir = Path(f"{args.result_dir}/{args.model}/trt_profiling")
    trt_profile_dir.mkdir(exist_ok=True, parents=True)
    trt_infer.save_engine_info(trt_profile_dir)
    trt_infer.save_layer_wise_profiling(trt_profile_dir)

    # Convert milliseconds to seconds
    timings = [cuda_call(cudart.cudaEventElapsedTime(s, e)) * 1.0e-3 for s, e in zip(start_events, end_events)]
    avg_throughput = args.input_shape[0] / np.mean(timings)
    results = BenchmarkMetrics(
        trt_load_time=trt_load_time,  # in seconds
        latencies=timings,  # in seconds
        avg_throughput=avg_throughput,
        avg_latency=np.mean(timings),  # in seconds
    )
    return results
