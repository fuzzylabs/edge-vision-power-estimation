"""Data preprocessing module."""

import shutil
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from tqdm import tqdm

from data_preparation.io_utils import read_json_file, read_log_file


# Test 
class MetricsByCycle(TypedDict):
    """Metrics collected by an inference cycle for a given layer."""

    cycle: int
    layer_name: str
    layer_type: str
    layer_power_including_idle_power_micro_watt: float | None
    layer_power_excluding_idle_power_micro_watt: float | None
    layer_run_time: float


def map_layer_name_to_type(trt_engine_info: dict) -> dict:
    """This function produce a mapping of layer type by the name of the layer.

    This is to help us identify which layer we are interested in
    in the `trt_layer_latency.json` since there is only layer name
    and not type in there.

    Args:
        trt_engine_info: the `trt_engine_info.json` file

    Returns:
        Dictionary of layer name to layer type.
    """
    return {
        layer["Name"]: layer["LayerType"]
        for layer in tqdm(
            trt_engine_info["Layers"], desc="Mapping layer name to layer type"
        )
    }


def parse_timestamp(timestamp: str) -> datetime:
    """Parse str to a datetime object.

    This will allow us to compare time before and after.

    Args:
        timestamp: timestamp str

    Returns:
        parsed timestamp.
    """
    return datetime.strptime(timestamp, "%Y%m%d-%H:%M:%S.%f")


class DataPreprocessor:
    """Data preprocessing class.

    For each model, there are 3 files
    1. Power measurement (*_power_log.json)
    2. Runtime measurement (trt_layer_latency.json)
    3. TensorRT engine information and other.

    Using this class, a CSV is created for each model with contents
    inference_cycle, layer_name, layer_type, ..., power_consumed, runtime_taken
    """

    def __init__(self, idle_power_log_path: Path, result_dir: Path):
        self.avg_idle_power = self._get_average_idling_power(idle_power_log_path)
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def _get_average_idling_power(self, file_path: Path) -> float:
        """Get the average idling power measure from json file.

        Args:
            file_path: Path to idle power log file.

        Returns:
            Average idling power (recorded in micro-watt).
        """
        print("Getting average idling power...")
        idling_power_log = read_json_file(file_path)
        return float(idling_power_log["avg_idle_power"])

    def preprocess_power_log(self, file_path: Path) -> list[tuple]:
        """Convert each power log entry to (datetime, power).

        Args:
            file_path: Path to model power log file

        Returns:
            Processed log for easy accessing the timestamp and power.
        """
        power_log = read_log_file(file_path)
        processed_log = []

        for entry in tqdm(power_log, desc="Preprocessing power log"):
            parts = entry.strip().split(",")
            timestamp = parse_timestamp(parts[0])
            voltage = float(parts[1])
            current = float(parts[2])
            processed_log.append((timestamp, voltage * current))

        return processed_log

    def compute_latency_start_end_times(
        self, trt_layer_latency: dict[str, list[list[float, str]]]
    ) -> list[tuple]:
        """Calculate start and end time for each layer.

        We use the start and end time for each layer to
        get the power values between the start and end time of layer inference.

        Example:
            Suppose we have a following trt latency data
            {
            layer_1 : [[5, 00:10]],
            layer_2 : [[2, 00:14]]
            }

            This function will calculate start time using end time and latency.
            layer_1
            start_time: 00:05
            end_time: 00:10
            execution_duration: 5

            layer_2
            start_time: 00:12
            end_time: 00:14
            execution_duration: 2

        Args:
            trt_layer_latency: Dictionary containing latency data for each layer.

        Returns:
            List of tuples (cycle, start_time, end_time, duration, layer_name).
        """
        latency_data = defaultdict(list)

        for layer_name, layer_times in tqdm(
            trt_layer_latency.items(), desc="Preprocessing latency data"
        ):
            for cycle, (execution_duration, execution_end_time) in enumerate(
                layer_times
            ):
                end_timestamp = parse_timestamp(execution_end_time)
                duration = timedelta(milliseconds=execution_duration)
                start_timestamp = end_timestamp - duration
                latency_data[cycle].append(
                    (
                        cycle,
                        start_timestamp,
                        end_timestamp,
                        execution_duration,
                        layer_name,
                    )
                )

        latency_data = [
            entry for cycle_data in latency_data.values() for entry in cycle_data
        ]

        return latency_data

    def compute_layer_metrics_by_cycle(
        self,
        power_log_path: Path,
        trt_layer_latency_path: Path,
        trt_engine_info_path: Path,
    ) -> list[MetricsByCycle]:
        """Computes and aggregates power and runtime metrics for each layer within a processing cycle.

        For each iteration cycle and for each layer,
        we get the corresponding power values in the start and
        end time of layer inference for that layer.
        The average power value is considered as the power
        consumed for that iteration and that layer.

        Args:
            power_log_path: Path to model power log file
            trt_layer_latency_path: Path to tensorrt layer latency file
            trt_engine_info_path: Path to tensorrt engine info file
        Returns:
            list[MetricsByCycle]: A list of dictionaries, each representing metrics for a specific layer.
        """
        print("Computing layer metrics...")
        power_logs = self.preprocess_power_log(power_log_path)
        power_logs_iterator = iter(power_logs)

        # Preprocess and sort latency data by start time
        trt_layer_latency = read_json_file(trt_layer_latency_path)
        latency_data = self.compute_latency_start_end_times(trt_layer_latency)
        trt_engine_info = read_json_file(trt_engine_info_path)
        layer_name_type_mapping = map_layer_name_to_type(trt_engine_info)

        metrics_by_cycle = []

        power_log = next(power_logs_iterator)

        def calculate_metrics(cycle: int, layer_name: str, layer_type: str, power_measurements: list[float], execution_duration: float) -> MetricsByCycle:
            """Calculate metrics for each cycle."""
            # Calculate average power if measurements exist
            avg_layer_power = (
                sum(power_measurements) / len(power_measurements)
                if power_measurements
                else None
            )

            avg_layer_power_excluding_idle = (avg_layer_power - self.avg_idle_power if avg_layer_power else None)

            return {
                "cycle": cycle + 1,
                "layer_name": layer_name,
                "layer_type": layer_type,
                "layer_power_including_idle_power_micro_watt": avg_layer_power,
                "layer_power_excluding_idle_power_micro_watt": avg_layer_power_excluding_idle,
                "layer_run_time": execution_duration,
            }


        for (
            cycle,
            start_timestamp,
            end_timestamp,
            execution_duration,
            layer_name,
        ) in tqdm(latency_data, desc="Mapping power to layer"):
            layer_type = layer_name_type_mapping.get(layer_name, "Unknown")
            layer_power_measurements = []

            try:
                # "Scroll" to the start time stamp if we are not there yet
                while (
                    power_log[0] < start_timestamp
                ):
                    power_log = next(power_logs_iterator)

                # Collect power measurements within start and end timestamp
                while (
                    power_log[0] <= end_timestamp
                ):
                    layer_power_measurements.append(power_log[1])
                    power_log = next(power_logs_iterator)

            except StopIteration:
                break
            finally:
                # Append the results for this layer and cycle
                metrics_by_cycle.append(calculate_metrics(
                    cycle,
                    layer_name,
                    layer_type,
                    layer_power_measurements,
                    execution_duration,
                ))


        return metrics_by_cycle

    def save_result_to_csv(
        self, metrics_by_cycle: list[dict[str, Any]], model_name: str
    ) -> None:
        """Save the power used and run time of individual layers to a csv.

        Args:
            metrics_by_cycle: List of dictionaries, each representing a cycle's data for a layer,
                including cycle number, layer name, type, power, and runtime.
            model_name: Name of the model
        """
        result_model_dir = f"{self.result_dir}/{model_name}"
        # Create directory if it does not exist
        Path(result_model_dir).mkdir(exist_ok=True, parents=True)
        filename = "power_runtime_mapping_layerwise.csv"

        df = pd.DataFrame.from_dict(metrics_by_cycle)
        df.to_csv(f"{result_model_dir}/{filename}", index=False)
        print(f"Metric results save to {self.result_dir}/{model_name}/{filename}")

    def copy_trt_engine_to_target_dir(
        self, model_name: str, trt_engine_info_path: Path
    ) -> None:
        """Copy tensorrt engine info file to target directory.

        Args:
            model_name: Name of the model
            trt_engine_info_path: Path to tensorrt engine info file
        """
        result_model_dir = f"{self.result_dir}/{model_name}"
        # Create directory if it does not exist
        Path(result_model_dir).mkdir(exist_ok=True, parents=True)
        shutil.copy2(src=trt_engine_info_path, dst=result_model_dir)
