"""Utility used by tensorrt backend."""

import json
from pathlib import Path
import tensorrt as trt
from datetime import datetime


class CustomProfiler(trt.IProfiler):
    """Custom Profiler for logging layer-wise latency."""

    def __init__(self):
        trt.IProfiler.__init__(self)
        self.layers = {}

    def report_layer_time(self, layer_name, ms):
        if layer_name not in self.layers:
            self.layers[layer_name] = []

        current_time = datetime.now().strftime("%Y%m%d-%H:%M:%S.%f")  # Time with seconds and microseconds
        self.layers[layer_name].append((ms, current_time))


def save_layer_wise_profiling(mod, layer_latency_dir: str) -> None:
    """Save layer wise latency information to json file.

    Args:
        mod: PythonTorchTensorRTModule
        layer_latency_dir: Path to save latency times.
    """
    layer_latency = mod.context.profiler.layers
    layer_latency_path = Path(layer_latency_dir) / "trt_layer_latency.json"
    with open(layer_latency_path, "w") as fp:
        json.dump(layer_latency, fp, indent=4)


def save_engine_info(mod, engine_dir) -> None:
    """Save information of the engine in json format.

    Args:
        mod: PythonTorchTensorRTModule
        engine_dir: Path to save engine information.
    """
    inspector = mod.engine.create_engine_inspector()
    engine_json: str = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    # Save engine information
    engine_json_path = Path(engine_dir) / "trt_engine_info.json"
    with open(engine_json_path, "w") as fp:
        json.dump(json.loads(engine_json), fp, indent=4)