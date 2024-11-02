"""This script maps the power to the individual layer.

Example command:
python map_power_to_layers.py \
    --power-log-path results/5000_cycles_power_log_20241029-144359.log \
    --trt-layer-latency-path results/mobilenet_v2/trt_profiling/trt_layer_latency.json \
    --trt-engine-info results/mobilenet_v2/trt_profiling/trt_engine_info.json \
    --result-dir results

"""
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import argparse
from collections import defaultdict


def map_layer_name_to_type(trt_engine_info: dict) -> dict:
    """
    This function produce a mapping of layer type by the name of the layer.

    This is to help us identify which layer we are interested in
    in the `trt_layer_latency.json` since there is only layer name
    and not type in there.

    Args:
        trt_engine_info: the `trt_engine_info.json` file

    Returns:
        Dictionary of layer name to layer type.
    """
    return {layer["Name"]: layer["LayerType"] for layer in trt_engine_info["Layers"]}


def parse_timestamp(timestamp: str) -> datetime:
    """
    Parse str to a datetime object.

    This will allow us to compare time before and after.

    Args:
        timestamp: timestamp str

    Returns:
        parsed timestamp.
    """
    return datetime.strptime(timestamp, "%Y%m%d-%H:%M:%S.%f")


def preprocess_power_log(power_log: list[str]) -> list[tuple]:
    """
    Convert each power log entry to (datetime, power).

    Args:
        power_log: a read power log file

    Returns:
        processed log for easy accessing the timestamp and power.
    """
    processed_log = []

    for entry in power_log:
        parts = entry.strip().split(',')
        timestamp = parse_timestamp(parts[0])
        voltage = float(parts[1])
        current = float(parts[2])
        processed_log.append((timestamp, voltage, current))

    return processed_log


def preprocess_latency_data(trt_layer_latency: dict[str, list[list[float, str]]]) -> list[tuple]:
    """
    Preprocess and sort latency data by adjusted start time. See below for what adjusted start time mean.

    Adjust start time:
        Each layer (except the first in each cycle) starts at the previous layer's end time or its own recorded start time, whichever is later.

        Check this thread for more detail:
            https://fuzzy-labs.slack.com/archives/C07DTP3RGLA/p1730470951456389?thread_ts=1730281446.171379&cid=C07DTP3RGLA

    Args:
        trt_layer_latency: Dictionary containing latency data for each layer.

    Returns:
        List of tuples with (cycle, start_time, end_time, duration, layer_name), sorted by start time.
    """
    latency_data = defaultdict(list)

    for layer_name, layer_times in trt_layer_latency.items():
        for cycle, (execution_duration, execution_start_time) in enumerate(layer_times):
            start_timestamp = parse_timestamp(execution_start_time)
            duration = timedelta(milliseconds=execution_duration)
            
            # If not first layer.
            if latency_data[cycle]:
                previous_layer_end_time = latency_data[cycle][-1][2]
                # Adjust start time if needed
                start_timestamp = max(previous_layer_end_time, start_timestamp)

            # Calculate end time based on adjusted start time
            end_timestamp = start_timestamp + duration
            latency_data[cycle].append((cycle, start_timestamp, end_timestamp, execution_duration, layer_name))

    sorted_latency_data = [entry for cycle_data in latency_data.values() for entry in cycle_data]
    sorted_latency_data.sort(key=lambda x: x[1])

    return sorted_latency_data


def compute_layer_metrics_by_cycle(
    trt_layer_latency: dict[str, list[list[float, str]]],
    power_log: list[str],
    trt_engine_info: dict[str, str]
) -> list[dict[str, any]]:
    
    power_logs = preprocess_power_log(power_log)  # Ensure this function returns [(timestamp, power1, power2), ...]

    # Preprocess and sort latency data by start time
    sorted_latency_data = preprocess_latency_data(trt_layer_latency)
    layer_name_type_mapping = map_layer_name_to_type(trt_engine_info)

    metrics_by_cycle = []
    power_index = 0

    for cycle, _, end_timestamp, execution_duration, layer_name in sorted_latency_data:
        layer_type = layer_name_type_mapping.get(layer_name, "Unknown")
        layer_power_measurements = []

        # Collect power measurements within start and end timestamp
        while power_index < len(power_logs) and power_logs[power_index][0] <= end_timestamp:
            layer_power_measurements.append(power_logs[power_index][1] * power_logs[power_index][2])
            power_index += 1
        
        # Calculate average power if measurements exist
        avg_layer_power = sum(layer_power_measurements) / len(layer_power_measurements) if layer_power_measurements else 0.0

        # Append the results for this layer and cycle
        metrics_by_cycle.append({
            "cycle": cycle + 1,
            "layer_name": layer_name,
            "layer_type": layer_type,
            "average_power_micro_watt": avg_layer_power,
            "layer_run_time": execution_duration
        })

        # Adjust `power_log_index` to backtrack by 1 to re-evaluate on the next layer if needed
        # This is because next layer start time could be equal to prev layer end time.
        power_index = max(0, power_index - 1)

    return metrics_by_cycle


def save_result_to_csv(
    metrics_by_cycle: list[dict[str, any]],
    args: argparse.Namespace,
) -> None:
    """
    Save the power used and run time of individual layers to a csv.

    Args:
        metrics_by_cycle: List of dictionaries, each representing a cycle's data for a layer,
            including cycle number, layer name, type, power, and runtime.
        args: Arguments from CLI.
    """
    df = pd.DataFrame.from_dict(metrics_by_cycle)

    df.to_csv(f"{args.result_dir}/metrics_by_cycle.csv")


def read_log_files(
    args: argparse.Namespace,
) -> tuple[list[str], dict]:
    """
    Reads power log and TRT layer latency data from specified file paths.

    Args:
        args: Arguments from CLI.

    Returns:
        A tuple containing the power logs, layer latency and layer info.
    """
    with open(args.power_log_path) as power_log_file, open(
        args.trt_layer_latency_path) as trt_layer_latency_file, open(
            args.trt_engine_info) as trt_engine_info_file:
        return power_log_file.readlines(), json.load(trt_layer_latency_file), json.load(trt_engine_info_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Power Mapping With Logs Collected on Jetson",
        description="Map power usage data during inference cycles for ImageNet pretrained CNN models."
    )
    parser.add_argument(
        "--power-log-path",
        type=Path,
        help="The path of where the power log file are stored."
        "The file should have the following format:"
        "n_cycles_power_log_timestamp.log",
    )
    parser.add_argument(
        "--trt-layer-latency-path",
        type=Path,
        help="The path of where the trt layer latency profile result are stored."
        "The file should be stored under the `trt_profiling` directory",
    )
    parser.add_argument(
        "--trt-engine-info",
        type=Path,
        help="The path of where the trt engine info are stored."
        "The file should be stored under the `trt_profiling` directory",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="results",
        help="The directory to save the csv."
    )
    args = parser.parse_args()

    power_log, trt_layer_latency, trt_engine_info = read_log_files(args)

    # layer_type_mapping = map_layer_name_to_type(trt_engine_info)
    metrics_by_cycle = compute_layer_metrics_by_cycle(
        trt_layer_latency=trt_layer_latency,
        power_log=power_log,
        trt_engine_info=trt_engine_info,   
    )

    save_result_to_csv(
        metrics_by_cycle=metrics_by_cycle,
        args=args
    )
