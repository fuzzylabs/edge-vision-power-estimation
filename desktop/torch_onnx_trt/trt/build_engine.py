"""Build TensorRT engine from ONNX model.

To run the build engine script:
    python trt/build_engine.py \
        --onnx models/alexnet/alexnet.onnx
"""

from pathlib import Path
import argparse
import os
import time
import tensorrt as trt
import sys
from utils import save_timing_cache, setup_timing_cache


# Modified from https://github.com/NVIDIA/TensorRT/blob/main/samples/python/efficientnet/build_engine.py
class TRTEngineBuilder:
    """Parses an ONNX graph and builds a TensorRT engine from it."""

    def __init__(self, workspace: int):
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#build_engine_python
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace * (2**30)
        )

        self.batch_size = None
        self.network = None
        self.parser = None
        self.dtype = None

    def create_network(self, onnx_path: str, batch_size: int, dynamic_batch_size=None):
        """Parse the ONNX graph and create the corresponding TensorRT network definition.

        Args:
            onnx_path: The path to the ONNX graph to load.
            batch_size: Static batch size to build the engine with.
            dynamic_batch_size: Dynamic batch size to build the engine with, if given,
                        pass as a comma-separated string or int list as MIN,OPT,MAX
        """
        self.network = self.builder.create_network(0)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        self.onnx_path = os.path.realpath(onnx_path)
        with open(self.onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                print("Failed to load ONNX file: {}".format(self.onnx_path))
                for error in range(self.parser.num_errors):
                    print(self.parser.get_error(error))
                sys.exit(1)

        print("Network Description")

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        input_dtypes, output_dtypes = [], []
        profile = self.builder.create_optimization_profile()
        dynamic_inputs = False
        for input in inputs:
            print(
                f"Input '{input.name}' with shape {input.shape} and dtype {input.dtype}"
            )
            input_dtypes.append(input.dtype)
            # If dynamic shapes are passed create a optimization profile
            # for all the dynamic shapes
            if input.shape[0] == -1:
                dynamic_inputs = True
                if dynamic_batch_size:
                    if type(dynamic_batch_size) is str:
                        dynamic_batch_size = [
                            int(v) for v in dynamic_batch_size.split(",")
                        ]
                    assert len(dynamic_batch_size) == 3
                    min_shape = [dynamic_batch_size[0]] + list(input.shape[1:])
                    opt_shape = [dynamic_batch_size[1]] + list(input.shape[1:])
                    max_shape = [dynamic_batch_size[2]] + list(input.shape[1:])
                    profile.set_shape(input.name, min_shape, opt_shape, max_shape)
                    print(
                        f"Input '{input.name}' Optimization Profile with "
                        f"shape MIN {min_shape} / OPT {opt_shape} / MAX {max_shape} and dtype {input.dtype}"
                    )
                else:
                    shape = [batch_size] + list(input.shape[1:])
                    profile.set_shape(input.name, shape, shape, shape)
                    print(
                        f"Input '{input.name}' Optimization Profile with shape {shape} and dtype {input.dtype}"
                    )
        if dynamic_inputs:
            self.config.add_optimization_profile(profile)

        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]
        for output in outputs:
            print(
                f"Output '{output.name}' with shape {output.shape} and dtype {output.dtype}"
            )
            output_dtypes.append(input.dtype)

        assert set(input_dtypes) == set(output_dtypes)
        self.dtype = set(input_dtypes)

    def extract_model_info(self, path_str):
        path = Path(path_str)
        filename = path.stem
        parts = filename.split("_")
        model_name = "_".join(parts[:-1])
        precision = parts[-1]
        return model_name, precision

    def create_engine(self, optimization_level: int, timing_cache: bool):
        """Build the TensorRT engine and serialize it to disk.

        Args:
            optimization_level: Builder optimization 0-5, higher levels imply longer build time
            save_timing_cache: Whether to save timing cache to a file.
                This avoid rebuilding the tensorrt engine for each run.
        """
        start = time.perf_counter()
        model_dir = Path(self.onnx_path).parent
        model_name, precision = self.extract_model_info(Path(self.onnx_path))
        engine_path = f"{model_dir}/{model_name}_{precision}.engine"
        print(f"Building {precision} Engine in {engine_path}")

        if timing_cache:
            timing_cache_path = f"{model_dir}/{model_name}_{precision}_timing.cache"
            print("Reading timing cache from file: {:}".format(timing_cache_path))
            setup_timing_cache(self.config, timing_cache_path)

        self.config.builder_optimization_level = optimization_level
        self.config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        # TODO: Add support for int8 quantization
        # Building engine for int8 precision requires explicit
        # quantization. Ref: https://github.com/NVIDIA/TensorRT/issues/4095
        if self.dtype == trt.DataType.HALF:
            if not self.builder.platform_has_fast_fp16:
                print("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)

        if self.dtype == trt.DataType.FLOAT:
            self.config.set_flag(trt.BuilderFlag.FP32)

        engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        if engine_bytes is None:
            print("Failed to create engine")
            sys.exit(1)

        if timing_cache:
            print("Serializing timing cache to file: {:}".format(timing_cache_path))
            save_timing_cache(self.config, timing_cache_path)

        with open(engine_path, "wb") as f:
            print("Serializing engine to file: {:}".format(engine_path))
            f.write(engine_bytes)

        print(f"Time taken to build TRT engine {time.perf_counter()-start:.3f} sec")


def main(args: argparse.Namespace) -> None:
    """Main entrypoint to build TRT engine.

    Args:
        args: Arguments from CLI.
    """
    builder = TRTEngineBuilder(args.workspace)
    builder.create_network(args.onnx, args.batch_size, args.dynamic_batch_size)
    builder.create_engine(args.optimization_level, args.timing_cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", help="The input ONNX model file to load")
    parser.add_argument(
        "-b",
        "--batch_size",
        default=1,
        type=int,
        help="The static batch size to build the engine with, default: 1",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=5,
        help="Builder optimization 0-5, higher levels imply longer build time, "
        "searching for more optimization options.",
    )
    parser.add_argument(
        "-w",
        "--workspace",
        default=8,
        type=int,
        help="The max memory workspace size to allow in Gb, default: 8",
    )
    parser.add_argument(
        "-d",
        "--dynamic_batch_size",
        default=None,
        help="Enable dynamic batch size by providing a comma-separated MIN,OPT,MAX batch size, "
        "if this option is set, --batch_size is ignored, example: -d 1,16,32, "
        "default: None, build static engine",
    )
    parser.add_argument(
        "--timing-cache",
        action="store_true",
        help="Specify to timing cache to file to avoid rebuilding the engine.",
    )
    args = parser.parse_args()
    main(args)
