"""Utility functions for inspecting power profiles."""

from collections import defaultdict
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_power_log(log_values: list[str]) -> list[tuple]:
    """Convert each power log entry to (datetime, power).

    Args:
        log_values: List of timestamp, current and voltage as string.

    Returns:
        Processed log for easy accessing the timestamp and power.
    """
    result = []
    for entry in tqdm(log_values):
        parts = entry.strip().split(",")
        timestamp = datetime.strptime(parts[0], "%Y%m%d-%H:%M:%S.%f")
        voltage = float(parts[1])
        current = float(parts[2])
        result.append((timestamp, voltage * current))
    return result


def parse_latencies(
    trt_layer_latency: dict[str, list[list[float, str]]],
) -> defaultdict:
    """Calculate start and end time for each layer.

    Args:
        trt_layer_latency: Dictionary containing latency data for each layer.

    Returns:
        Dictionary of cytcles where each cycle is
        a list of tuples (cycle, start_time, end_time, duration, layer_name).
    """
    latency_data = defaultdict(list)
    for layer_name, layer_times in tqdm(trt_layer_latency.items()):
        for cycle, (execution_duration, execution_end_time) in enumerate(layer_times):
            end_timestamp = datetime.strptime(execution_end_time, "%Y%m%d-%H:%M:%S.%f")
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
    return latency_data


def plot_inference_cycle_power(
    cycle: int, power_data: list[tuple], latency_data: dict[list[tuple]], delay: int
) -> None:
    """Plot power profile for a given inference cycle.

    It calculates power indexes in the power log corresponding to
    start of inference, end of inference and end of the delay.
    It plots a power profile using these values.

    Args:
        cycle: Inference cycle
        power_data: Data containing (timestamp, power) values for entire experiment.
        latency_data: Latency data for all inference cycles.
            It contains start time, end time, duration, name,
            inference cycle for each layer in the model.
        delay: Delay in seconds in between inference calls
    """
    curr_cycle = latency_data[cycle]
    first_cycle_first_layer_start_time = curr_cycle[0][1]
    first_cycle_last_layer_end_time = curr_cycle[-1][2]
    next_cycle_with_delay = curr_cycle[-1][2] + timedelta(0, delay)
    print(f"{first_cycle_first_layer_start_time} -> {first_cycle_last_layer_end_time}")
    print(next_cycle_with_delay)
    print(
        f"Time taken by inference cycle {cycle}: {first_cycle_last_layer_end_time - first_cycle_first_layer_start_time}"
    )

    i, flag = 0, False
    while True:
        timestamp = power_data[i][0]
        if timestamp >= first_cycle_first_layer_start_time:
            start_index = i
            while timestamp <= next_cycle_with_delay:
                timestamp = power_data[i][0]
                i += 1
                flag = True
            end_delay_index = i
            break
        i += 1
        if flag:
            break
    i = start_index
    timestamp = power_data[i][0]
    while timestamp <= first_cycle_last_layer_end_time:
        timestamp = power_data[i][0]
        i += 1
        flag = True
    end_index = i
    print(f"Iteration: {cycle}, Power index: {start_index} -> {end_index}")
    print(end_delay_index)

    fig = plt.figure()
    plt.plot(
        range(start_index, end_delay_index),
        [power for (_, power) in power_data[start_index:end_delay_index]],
    )
    plt.axvline(start_index, color="green", label="inference start")
    plt.axvline(end_index, color="blue", label="inference end")
    plt.axvline(end_delay_index, color="purple", label="delay end")
    plt.legend()
    plt.show()
    print("Power values:")
    print(f"Inference start: {power_data[start_index][1]}")
    print(f"Inference end: {power_data[end_index][1]}")
    print(f"Delay end: {power_data[end_delay_index][1]}")
    print("-" * 100)
