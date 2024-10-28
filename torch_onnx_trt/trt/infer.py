import tensorrt as trt
from cuda import cudart
from .runtime import memcpy_device_to_host, memcpy_host_to_device, cuda_call
import numpy as np
from pathlib import Path
import json


class CustomProfiler(trt.IProfiler):
    """Custom Profiler for logging layer-wise latency."""

    def __init__(self):
        trt.IProfiler.__init__(self)
        self.layers = {}

    def report_layer_time(self, layer_name, ms):
        if layer_name not in self.layers:
            self.layers[layer_name] = []

        self.layers[layer_name].append(ms)


class TensorRTInfer:
    """Implements inference for the TensorRT engine."""

    def __init__(self, engine_path: str, enable_profiler: bool) -> None:
        """Constructor for TensorRTInfer class.

        Args:
            engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        self.setup_input_output()

        if enable_profiler:
            self.enable_profiling()

    def setup_input_output(self) -> None:
        """Setup input, output bindings from TensorRT engine."""
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda_call(cudart.cudaMalloc(size))
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
            }
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def enable_profiling(self, profiler: "trt.IProfiler" = None) -> None:
        """Enable TensorRT profiling.

        TensorRT will report time spent on each layer in stdout for each forward run.
        """
        if not self.context.profiler:
            self.context.profiler = CustomProfiler() if profiler is None else profiler
            # self.context.profiler = trt.Profiler() if profiler is None else profiler

    def save_engine_info(self, engine_dir) -> None:
        """Save information of the engine in json format."""
        inspector = self.engine.create_engine_inspector()
        engine_json: str = inspector.get_engine_information(
            trt.LayerInformationFormat.JSON
        )
        # Save engine information
        engine_json_path = Path(engine_dir) / "trt_engine_info.json"
        with open(engine_json_path, "w") as fp:
            json.dump(json.loads(engine_json), fp, indent=4)

    def save_layer_wise_profiling(self, layer_latency_dir: str) -> None:
        layer_latency = self.context.profiler.layers
        # Save layer wise latency information
        layer_latency_path = Path(layer_latency_dir) / "trt_layer_latency.json"
        with open(layer_latency_path, "w") as fp:
            json.dump(layer_latency, fp, indent=4)

    def input_spec(self):
        """Get the specs for the input tensor of the network. Useful to prepare memory allocations.

        Returns:
            Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]

    def output_spec(self):
        """Get the specs for the output tensor of the network. Useful to prepare memory allocations.

        Returns:
            Two items, the shape of the output tensor and its (numpy) datatype.
        """
        return self.outputs[0]["shape"], self.outputs[0]["dtype"]

    def infer(self, batch):
        """Execute inference on a batch of images.

        Args:
            batch: A numpy array holding the image batch.

        Returns:
            Output as numpy arrays for each batch image
        """
        # Prepare the output data
        output = np.zeros(*self.output_spec())

        # Process I/O and execute the network
        memcpy_host_to_device(self.inputs[0]["allocation"], np.ascontiguousarray(batch))
        self.context.execute_v2(self.allocations)
        memcpy_device_to_host(output, self.outputs[0]["allocation"])
        return output
