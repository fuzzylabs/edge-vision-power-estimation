"""Utility functions for inspecting power profiles."""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import TypedDict

import matplotlib.pyplot as plt
from tqdm import tqdm


class PytorchLayerLatency(TypedDict):
    """Measured layer latencies in Pytorch for a single inference cycle.

    Keyed by the layer name.
    """

    start_time: float
    """Layer execution UNIX timestamp in seconds."""

    elapsed_time: float
    """Layer execution elapsed time in milliseconds."""


class MetricsByCycle(TypedDict):
    """Metrics collected by an inference cycle for a given layer."""

    cycle: int
    layer_name: str
    layer_type: str
    layer_power_including_idle_power_micro_watt: float | None
    layer_power_excluding_idle_power_micro_watt: float | None
    layer_run_time: float


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
    pytorch_layer_latency: list[dict[str, PytorchLayerLatency]],
) -> defaultdict:
    """Calculate start and end time for each layer.

    Args:
        trt_layer_latency: Dictionary containing latency data for each layer.

    Returns:
        Dictionary of cytcles where each cycle is
        a list of tuples (cycle, start_time, end_time, duration, layer_name).
    """
    latency_data = defaultdict(list)

    for cycle, cycle_layer_latency in tqdm(
        enumerate(pytorch_layer_latency), desc="Computing layer execution times"
    ):
        for layer_name, layer_latency in cycle_layer_latency.items():
            start_timestamp = datetime.fromtimestamp(layer_latency["start_time"])
            duration = timedelta(milliseconds=layer_latency["elapsed_time"])
            end_timestamp = start_timestamp + duration
            latency_data[cycle].append(
                (
                    cycle,
                    start_timestamp,
                    end_timestamp,
                    layer_latency["elapsed_time"],
                    layer_name,
                )
            )
    return latency_data


def compute_layer_metrics_by_cycle(
    power_logs, latency_data, avg_idle_power
) -> list[MetricsByCycle]:
    """Computes and aggregates power and runtime metrics for each layer within a processing cycle.

    For each iteration cycle and for each layer,
    we get the corresponding power values in the start and
    end time of layer inference for that layer.
    The average power value is considered as the power
    consumed for that iteration and that layer.

    Args:
        power_logs: Power logs for the inference
        latency_data: List of dictionary containing data for each inference cycle

    Returns:
        list[MetricsByCycle]: A list of dictionaries, each representing metrics for a specific layer.
    """
    print("Computing layer metrics...")
    power_logs_iterator = iter(power_logs)

    latency_data = [
        entry for cycle_data in latency_data.values() for entry in cycle_data
    ]

    metrics_by_cycle = []

    power_log = next(power_logs_iterator)

    def calculate_metrics(
        cycle: int,
        power_measurements: list[float],
        execution_duration: float,
    ) -> MetricsByCycle:
        """Calculate metrics for each cycle."""
        # Calculate average power if measurements exist
        avg_layer_power = (
            sum(power_measurements) / len(power_measurements)
            if power_measurements
            else None
        )

        avg_layer_power_excluding_idle = (
            avg_layer_power - avg_idle_power if avg_layer_power else None
        )

        return {
            "cycle": cycle + 1,
            "layer_power_including_idle_power_micro_watt": avg_layer_power,
            "layer_power_excluding_idle_power_micro_watt": avg_layer_power_excluding_idle,
            "layer_run_time": execution_duration,
        }

    for (
        cycle,
        start_timestamp,
        end_timestamp,
        execution_duration,
        _,
    ) in tqdm(latency_data, desc="Mapping power to layer"):
        layer_power_measurements = []

        try:
            # "Scroll" to the start time stamp if we are not there yet
            while power_log[0] < start_timestamp:
                power_log = next(power_logs_iterator)

            # Collect power measurements within start and end timestamp
            while power_log[0] <= end_timestamp:
                layer_power_measurements.append(power_log[1])
                power_log = next(power_logs_iterator)

        except StopIteration:
            break
        finally:
            # Append the results for this layer and cycle
            metrics_by_cycle.append(
                calculate_metrics(cycle, layer_power_measurements, execution_duration)
            )

    return metrics_by_cycle


def get_power_index(power_data, timestamp):
    index = 0
    while True:
        power_timestamp = power_data[index][0]
        if power_timestamp > timestamp:
            return index
        index += 1
