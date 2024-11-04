# Modified from : https://github.com/NVIDIA/TensorRT/blob/main/tools/experimental/trt-engine-explorer/utils/process_engine.py
from typing import Optional
from pathlib import Path
import argparse
import os
import subprocess
import tempfile
import tensorrt as trt
import time
import trex.archiving as archiving
from utils import generate_build_metadata, generate_profiling_metadata

DTYPE_MAPPING = {"float32": "fp32", "float16": "fp16", "bfloat16": "bfp16"}


def get_engine_path(onnx_path: str) -> str:
    """Get path to save TensorRT engine.

    Args:
        onnx_path: Path to ONNX file.

    Returns:
        Path to save TensorRT engine.
    """
    onnx_fname = os.path.basename(onnx_path).split(".")[0]
    outdir = os.path.dirname(onnx_path)
    engine_path = os.path.join(outdir, onnx_fname) + ".engine"
    return engine_path


def run_trtexec(trt_cmdline: list[str], writer) -> bool:
    """Run trtexec CLI using subprocess.

    Args:
        trt_cmdline: List of arguments containing `trtexec` CLI
        writer: Writer to log output from `trtexec` CLI

    Returns:
        A boolean to indicate if running `trtexec` was success or not.
    """
    success = False
    with writer:
        log_str = None
        try:
            log = subprocess.run(
                trt_cmdline,
                check=True,
                # Redirection
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
            success = True
            log_str = log.stdout
        except subprocess.CalledProcessError as err:
            log_str = err.output
        except FileNotFoundError as err:
            log_str = f"\nError: {err.strerror}: {err.filename}"
            print(log_str)
        writer.write(log_str)
    return success


def build_engine_cmd(
    args: argparse.Namespace,
    onnx_path: str,
    engine_path: str,
    timing_cache_path: str,
    save_path: str,
) -> tuple[list[str], str]:
    """_summary_

    Args:
        args: _description_
        onnx_path: _description_
        engine_path: _description_
        timing_cache_path: _description_
        save_path: _description_

    Returns:
        _description_
    """
    graph_json_fname = f"{save_path}.graph.json"
    cmd_line = [
        "trtexec",
        "--verbose",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--exportLayerInfo={graph_json_fname}",
        f"--timingCacheFile={timing_cache_path}",
    ]
    if trt.__version__ < "10.0":
        # nvtxMode=verbose is the same as profilingVerbosity=detailed, but backward-compatible
        cmd_line.append("--nvtxMode=verbose")
        cmd_line.append("--buildOnly")
        cmd_line.append("--workspace=8192")
    else:
        cmd_line.append("--profilingVerbosity=detailed")

    # Defaults to fp32
    # Supported flags for dtype : --fp16, --bf16,--int8, --fp8,--noTF32, and --best
    if args.dtype == "float16":
        cmd_line.append("--fp16")

    build_log_fname = f"{engine_path}.build.log"
    return cmd_line, build_log_fname


def profile_engine_cmd(
    args: argparse.Namespace, engine_path: str, timing_cache_path: str, save_path: str
):
    profiling_json_fname = f"{save_path}.profile.json"
    graph_json_fname = f"{save_path}.graph.json"
    timing_json_fname = f"{save_path}.timing.json"
    cmd_line = [
        "trtexec",
        "--verbose",
        "--noDataTransfers",
        "--useCudaGraph",
        "--dumpLayerInfo",
        "--dumpProfile",
        f"--warmUp={args.warmup}",  # duration in ms
        f"--iterations={args.runs}",
        # Profiling affects the performance of your kernel!
        # Always run and time without profiling.
        "--separateProfileRun",
        "--useSpinWait",
        f"--loadEngine={engine_path}",
        f"--exportTimes={timing_json_fname}",
        f"--exportProfile={profiling_json_fname}",
        f"--exportLayerInfo={graph_json_fname}",
        f"--timingCacheFile={timing_cache_path}",
    ]
    if trt.__version__ < "10.0":
        cmd_line.append("--nvtxMode=verbose")
    else:
        cmd_line.append("--profilingVerbosity=detailed")

    # Defaults to fp32
    # Supported flags for dtype : --fp16, --bf16,--int8, --fp8,--noTF32, and --best
    if args.dtype == "float16":
        cmd_line.append("--fp16")

    profile_log_fname = f"{engine_path}.profile.log"
    return cmd_line, profile_log_fname


def build_engine(
    args: argparse.Namespace,
    onnx_path: str,
    engine_path: str,
    save_path: str,
    tea: Optional[archiving.EngineArchive] = None,
) -> bool:
    print(f"Building the engine: {engine_path}")
    cmd_line, build_log_file = build_engine_cmd(
        args, onnx_path, engine_path, args.timing_cache_path, save_path
    )
    print(" ".join(cmd_line))

    writer = archiving.get_writer(tea, build_log_file)
    success = run_trtexec(cmd_line, writer)
    if success:
        print("\nSuccessfully built the engine.\n")
        build_md_json_fname = f"{save_path}.build.metadata.json"
        generate_build_metadata(build_log_file, build_md_json_fname, tea)
    else:
        print("\nFailed to build the engine.")
        print(f"See logfile in: {build_log_file}\n")


def profile_engine(
    args: argparse.Namespace,
    engine_path: str,
    save_path: str,
    tea: Optional[archiving.EngineArchive] = None,
) -> bool:
    print(f"Profiling the engine: {engine_path}")
    cmd_line, profile_log_file = profile_engine_cmd(
        args, engine_path, args.timing_cache_path, save_path
    )
    print(" ".join(cmd_line))

    writer = archiving.get_writer(tea, profile_log_file)
    success = run_trtexec(cmd_line, writer)

    if success:
        print("\nSuccessfully profiled the engine.\n")
        profiling_md_json_fname = f"{save_path}.profile.metadata.json"
        generate_profiling_metadata(profile_log_file, profiling_md_json_fname, tea)
    else:
        print("\nFailed to profile the engine.")
        print(f"See logfile in: {profile_log_file}\n")


def benchmark_trt(args: argparse.Namespace, onnx_path: str) -> None:
    """Benchmark TensorRT engine.

    Build and profile TensorRT engine from ONNX model using `trtexec`.

    Args:
        args: _description_
        onnx_path: _description_
    """
    engine_path = get_engine_path(onnx_path)
    if not args.timing_cache_path:
        args.timing_cache_path = os.path.join(
            tempfile.gettempdir(), f"{args.model}_engine_cache"
        )
    save_dir = f"{args.result_dir}/{args.model}"
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    save_path = f"{save_dir}/{args.model}_{DTYPE_MAPPING.get(args.dtype)}"

    st = time.perf_counter()
    build_engine(args, onnx_path, engine_path, save_path)
    print(f"\nTime spent building engine: {time.perf_counter()-st:.3f} sec\n")

    st = time.perf_counter()
    profile_engine(args, engine_path, save_path)
    print(f"\nTime spent profiling engine: {time.perf_counter()-st:.3f} sec\n")
