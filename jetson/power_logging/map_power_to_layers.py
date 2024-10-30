"""This script maps the power to the individual layer."""
import json
from datetime import datetime, timedelta
from bisect import bisect_left
from collections import defaultdict

LAYERS_OF_INTEREST = {'PointWiseV2', 'CaskConvolution', 'CaskPooling', 'CaskGemmConvolution'}


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


def get_corresponding_power_log_from_timestamp(
    execution_duration: float,
    execution_start_timestamp: str,
    processed_log: list[tuple[datetime, float, float]],
) -> list[tuple[float, float]]:
    """
    Finds log entries within the execution time window.

    Args:
        execution_duration: Duration in seconds.
        execution_start_timestamp: Start timestamp as string.
        processed_log: List of (datetime, voltage, current).

    Returns:
        List of (voltage, current) tuples within the time window.
    """
    start_timestamp = parse_timestamp(execution_start_timestamp)
    end_timestamp = start_timestamp + timedelta(seconds=execution_duration)

    timestamps = [entry[0] for entry in processed_log]
    # We don't want to search through the power log file every time O(n).
    # This is similar to binary search o(log n)
    # We get the index of the next closest or matching timestamp in our power log file.
    start_index = bisect_left(timestamps, start_timestamp)

    matching_logs = []
    for i in range(start_index, len(processed_log)):
        log_timestamp, voltage, current = processed_log[i]
        if log_timestamp > end_timestamp:
            break
        matching_logs.append((voltage, current))
    
    return matching_logs


def compute_average_power_used_from_logs(logs: list[tuple[float, float]]) -> float:
    """
    Calculates the average power from voltage and current log entries.

    Args:
        logs: List of (voltage, current) tuples.

    Returns:
        Average power used as float.
    """
    if not logs:
        return 0.0
    
    power_measurements = [voltage * current for voltage, current in logs]
    return sum(power_measurements) / len(power_measurements)


def get_layer_average_power_used(
    trt_layer_latency: dict,
    power_log: list[str],
) -> dict[str, float]:
    """
    Calculates average power usage for each layer.

    Args:
        trt_layer_latency: Dictionary of layer timings.
        power_log: Raw power log entries.

    Returns:
        Dictionary of layer names and their average power used.
    """
    processed_log = preprocess_power_log(power_log)
    layer_power_used_for_all_cycles = defaultdict(list)

    for layer_name, layer_times in trt_layer_latency.items():
        for time in layer_times:
            execution_duration = time[0]
            execution_start_time = time[1]

            logs = get_corresponding_power_log_from_timestamp(
                execution_duration,
                execution_start_time,
                processed_log,
            )

            avg_power = compute_average_power_used_from_logs(logs)
            layer_power_used_for_all_cycles[layer_name].append(avg_power)
    
    # Calculate the overall average power used for each layer
    res = {layer_name: sum(powers) / len(powers)
           for layer_name, powers in layer_power_used_for_all_cycles.items() if powers}
    
    return res


def read_log_files(
    power_log_path: str,
    trt_layer_latency_path: str
) -> tuple[list[str], dict]:
    """
    Reads power log and TRT layer latency data from specified file paths.

    Args:
        power_log_path: Path to the power log file.
        trt_layer_latency_path: Path to the TRT layer latency json file.

    Returns:
        A tuple containing the power logs, and layer latency.
    """
    with open(power_log_path) as power_log_file, open(
        trt_layer_latency_path) as trt_layer_latency_file:
        return power_log_file.readlines(), json.load(trt_layer_latency_file)


if __name__ == "__main__":
    power_log, trt_layer_latency = read_log_files(
        power_log_path="jetson/power_logging/results_for_testing/5_cycles_power_log_20241030-150646.log",
        trt_layer_latency_path="jetson/power_logging/results_for_testing/trt_layer_latency.json"
    )

    average_power_used_by_layer = get_layer_average_power_used(
        trt_layer_latency=trt_layer_latency,
        power_log=power_log
    )

    with open(f"jetson/power_logging/results_for_testing/average_power_used_by_layer.json", "w") as f:
        json.dump(average_power_used_by_layer, f, indent=4)