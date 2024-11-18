"""Data preprocessing module."""

import re
import shutil
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from data_preparation.io_utils import read_json_file, read_log_file
from tqdm import tqdm


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


class DataPreprocessing:
    """Data preprocessing class.

    For each model, there are 3 files
    1. Power measurement (*_cycles_power*.join)
    2. Runtime measurement (trt_layer_latency.json)
    3. TensorRT engine information and other.

    Using this class, a CSV is created for each model with contents
    inference_cycle, layer_name, layer_type, ..., power_consumed, runtime_taken
    """

    def __init__(self, idle_power_log_path: str | Path, result_dir: str | Path):
        self.avg_idle_power = self._get_average_idling_power(idle_power_log_path)
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def _get_average_idling_power(self, file_path: str | Path) -> float:
        """Get the average idling power measure from log.

        Args:
            file_path: Path to idle power log file.

        Returns:
            Average idling power (recorded in micro-watt).
        """
        print("Getting average idling power...")
        idling_power_log = read_log_file(file_path)
        return float(re.search(r"[\d.]+", idling_power_log[0]).group())

    def preprocess_power_log(self, file_path: str | Path) -> list[tuple]:
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

    def preprocess_latency_data(
        self,
        trt_layer_latency: dict[str, list[list[float, str]]],
    ) -> list[tuple]:
        """Preprocess latency data by adjusting and sorting layers based on their start times.

        Start Time Adjustment:
        For each layer (except the first in each cycle), adjust the start time
        to the later of its own recorded start time or the end time of the preceding layer.

        Example:
            Suppose layer 1 finishes at 00:10 and takes 5 seconds to execute, giving it an end time of 00:15.
            If layer 2 has a recorded start time of 00:13 (which is before layer 1 ends),
            adjust layer 2's start time to 00:15 to maintain the sequential flow,
            as only one layer can run at a time.

        Note:
            This discrepancy in start times is mostly likely an artifact from the layer profiler.

        Args:
            trt_layer_latency: Dictionary containing latency data for each layer.

        Returns:
            List of tuples (cycle, start_time, end_time, duration, layer_name), sorted by start time.
        """
        latency_data = defaultdict(list)

        for layer_name, layer_times in tqdm(
            trt_layer_latency.items(), desc="Preprocessing latency data"
        ):
            for cycle, (execution_duration, execution_start_time) in enumerate(
                layer_times
            ):
                start_timestamp = parse_timestamp(execution_start_time)
                duration = timedelta(milliseconds=execution_duration)

                # If not first layer.
                if latency_data[cycle]:
                    previous_layer_end_time = latency_data[cycle][-1][2]
                    # Adjust start time if needed
                    start_timestamp = max(previous_layer_end_time, start_timestamp)

                # Calculate end time based on adjusted start time
                end_timestamp = start_timestamp + duration
                latency_data[cycle].append(
                    (
                        cycle,
                        start_timestamp,
                        end_timestamp,
                        execution_duration,
                        layer_name,
                    )
                )

        sorted_latency_data = [
            entry for cycle_data in latency_data.values() for entry in cycle_data
        ]
        sorted_latency_data.sort(key=lambda x: x[1])

        return sorted_latency_data

    def compute_layer_metrics_by_cycle(
        self,
        power_log_path: str | Path,
        trt_layer_latency_path: str | Path,
        trt_engine_info_path: str | Path,
    ) -> list[dict[str, Any]]:
        """Computes and aggregates power and runtime metrics for each layer within a processing cycle.

        Args:
            power_log_path: Path to model power log file
            trt_layer_latency_path: Path to tensorrt layer latency file
            trt_engine_info_path: Path to tensorrt engine info file
        Returns:
            A list of dictionaries, each representing metrics for a specific layer.
        """
        print("Computing layer metrics...")
        power_logs = self.preprocess_power_log(power_log_path)

        # Preprocess and sort latency data by start time
        trt_layer_latency = read_json_file(trt_layer_latency_path)
        sorted_latency_data = self.preprocess_latency_data(trt_layer_latency)
        trt_engine_info = read_json_file(trt_engine_info_path)
        layer_name_type_mapping = map_layer_name_to_type(trt_engine_info)

        metrics_by_cycle = []
        power_index = 0

        for cycle, _, end_timestamp, execution_duration, layer_name in tqdm(
            sorted_latency_data, desc="Mapping power to layer"
        ):
            layer_type = layer_name_type_mapping.get(layer_name, "Unknown")
            layer_power_measurements = []

            # Collect power measurements within start and end timestamp
            while (
                power_index < len(power_logs)
                and power_logs[power_index][0] <= end_timestamp
            ):
                layer_power_measurements.append(power_logs[power_index][1])
                power_index += 1

            # Calculate average power if measurements exist
            avg_layer_power = (
                sum(layer_power_measurements) / len(layer_power_measurements)
                if layer_power_measurements
                else 0.0
            )

            # Append the results for this layer and cycle
            metrics_by_cycle.append(
                {
                    "cycle": cycle + 1,
                    "layer_name": layer_name,
                    "layer_type": layer_type,
                    "layer_power_including_idle_power_micro_watt": avg_layer_power,
                    "layer_power_excluding_idle_power_micro_watt": avg_layer_power
                    - self.avg_idle_power,
                    "layer_run_time": execution_duration,
                }
            )

            # Adjust `power_log_index` to backtrack by 1 to re-evaluate on the next layer if needed
            # This is because next layer start time could be equal to prev layer end time.
            power_index = max(0, power_index - 1)

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
        self, model_name: str, trt_engine_info_path: str | Path
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
