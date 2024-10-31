# Modified from : https://github.com/NVIDIA/TensorRT/blob/main/tools/experimental/trt-engine-explorer/utils/process_engine.py
from typing import Optional
import subprocess
import argparse
import tensorrt as trt
import os
import tempfile
import trex.archiving as archiving
from utils import generate_build_metadata, generate_profiling_metadata


def append_trtexec_args(trt_args: dict, cmd_line: list[str]):
    for arg in trt_args:
        cmd_line.append(f"--{arg}")


def get_engine_path(onnx_path: str) -> str:
    onnx_fname = os.path.basename(onnx_path).split(".")[0]
    outdir = os.path.dirname(onnx_path)
    engine_path = os.path.join(outdir, onnx_fname) + ".engine"
    return engine_path


def run_trtexec(trt_cmdline: list[str], writer):
    """_summary_

    Args:
        trt_cmdline: _description_
        writer: _description_

    Returns:
        _description_
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
    args: argparse.Namespace, onnx_path: str, engine_path: str, timing_cache_path: str
) -> tuple[list[str], str]:
    graph_json_fname = f"{engine_path}.graph.json"
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

    append_trtexec_args(args.trtexec, cmd_line)

    build_log_fname = f"{engine_path}.build.log"
    return cmd_line, build_log_fname


def profile_engine_cmd(
    args: argparse.Namespace, engine_path: str, timing_cache_path: str
):
    profiling_json_fname = f"{engine_path}.profile.json"
    graph_json_fname = f"{engine_path}.graph.json"
    timing_json_fname = f"{engine_path}.timing.json"
    cmd_line = [
        "trtexec",
        "--verbose",
        "--noDataTransfers",
        "--useCudaGraph",
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

    append_trtexec_args(args.trtexec, cmd_line)

    profile_log_fname = f"{engine_path}.profile.log"
    return cmd_line, profile_log_fname


def build_engine(
    args: argparse.Namespace,
    onnx_path: str,
    engine_path: str,
    tea: Optional[archiving.EngineArchive],
) -> bool:
    print(f"Building the engine: {engine_path}")
    cmd_line, build_log_file = build_engine_cmd(
        args, onnx_path, engine_path, args.timing_cache_path
    )
    print(" ".join(cmd_line))

    writer = archiving.get_writer(tea, build_log_file)
    success = run_trtexec(cmd_line, writer)
    if success:
        print("\nSuccessfully built the engine.\n")
        build_md_json_fname = f"{engine_path}.build.metadata.json"
        generate_build_metadata(build_log_file, build_md_json_fname, tea)
    else:
        print("\nFailed to build the engine.")
        print(f"See logfile in: {build_log_file}\n")
    return success


def profile_engine(
    args: argparse.Namespace,
    engine_path: str,
    tea: Optional[archiving.EngineArchive],
) -> bool:
    print(f"Profiling the engine: {engine_path}")
    cmd_line, profile_log_file = profile_engine_cmd(
        args, engine_path, args.timing_cache_path
    )
    print(" ".join(cmd_line))

    writer = archiving.get_writer(tea, profile_log_file)
    success = run_trtexec(cmd_line, writer)

    if success:
        print("\nSuccessfully profiled the engine.\n")
        profiling_md_json_fname = f"{engine_path}.profile.metadata.json"
        generate_profiling_metadata(profile_log_file, profiling_md_json_fname, tea)
    else:
        print("\nFailed to profile the engine.")
        print(f"See logfile in: {profile_log_file}\n")
    return success


def benchmark_trt(args, onnx_path):
    engine_path = get_engine_path(onnx_path)
    tea_name = f"{engine_path}.tea"
    if not args.timing_cache_path:
        args.timing_cache_path = os.path.join(
            tempfile.gettempdir(), f"{args.model}_engine_cache"
        )

    tea = archiving.EngineArchive(tea_name) if args.archive else None
    if tea:
        tea.open()
    build_success = build_engine(args, tea)
    profile_success = profile_engine(args, tea)
    if tea:
        tea.close()
    return build_success, profile_success
