"""This script maps the power to the individual layer.

python map_power_to_layers.py \
    --power-log-path results_for_testing/5_cycles_power_log_20241030-150646.log \
    --trt-layer-latency-path results_for_testing/trt_layer_latency.json \
    --trt-engine-info results_for_testing/trt_engine_info.json \
    --result-dir results_for_testing

"""
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import argparse


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
    Convert each power log entry to (datetime, voltage, current).

    Args:
        power_log: a read power log file

    Returns:
        processed log for easy accessing voltage and current.
    """
    processed_log = []

    for entry in power_log:
        parts = entry.strip().split(',')
        timestamp = parse_timestamp(parts[0])
        voltage = float(parts[1])
        current = float(parts[2])
        processed_log.append((timestamp, voltage, current))

    return processed_log


def compute_layer_metrics_by_cycle(
    trt_layer_latency: dict[str, list[list[float, str]]],
    power_log: list[str],
    trt_engine_info: dict[str, str]
) -> list[dict[str, any]]:
    """
    Calculates power usage and runtime for each layer in each cycle.

    Args:
        trt_layer_latency: Dictionary of layer timings.
        power_log: Raw power log entries.
        trt_engine_info: Dictionary of layer information with types.

    Returns:
        List of dictionaries, each representing a cycle's data for a layer,
        including cycle number, layer name, type, power, and runtime.
    """
    processed_log = preprocess_power_log(power_log)
    layer_name_type_mapping = map_layer_name_to_type(trt_engine_info)

    metrics_by_cycle = []
    num_power_logs = len(processed_log)

    for layer_name, layer_times in trt_layer_latency.items():
        layer_type = layer_name_type_mapping.get(layer_name, "Unknown")

        for cycle_index, (execution_duration, execution_start_time) in enumerate(layer_times):
            current_log_index = 0
            start_timestamp = parse_timestamp(execution_start_time)
            end_timestamp = start_timestamp + timedelta(milliseconds=execution_duration)

            # This stores the power measured point for the SAME cycle
            cycle_power_measurements = []

            while current_log_index < num_power_logs:
                log_timestamp, voltage, current = processed_log[current_log_index]
                # processed_log at current_log_index is exceed the execution window
                if log_timestamp > end_timestamp:
                    break
                # processed_log at current_log_index is within the execution window
                if log_timestamp >= start_timestamp:
                    cycle_power_measurements.append(voltage * current)
                current_log_index += 1

            avg_cycle_power = sum(cycle_power_measurements) / len(cycle_power_measurements) if cycle_power_measurements else 0.0
            metrics_by_cycle.append({
                "cycle": cycle_index + 1,
                "layer_name": layer_name,
                "layer_type": layer_type,
                "average_power": avg_cycle_power, # The is the average power of the same cycle.
                "layer_run_time": execution_duration
            })

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

    df.to_csv(f"{args.result_dir}/average_power_and_run_time.csv")


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

    metrics_by_cycle = compute_layer_metrics_by_cycle(
        trt_layer_latency=trt_layer_latency,
        power_log=power_log,
        trt_engine_info=trt_engine_info,   
    )

    save_result_to_csv(
        metrics_by_cycle=metrics_by_cycle,
        args=args
    )
