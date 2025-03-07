"""Run inference for PyTorch model."""

from collections import defaultdict
from pathlib import Path
from statistics import mean

import pandas as pd
from rich import print
from rich.console import Console
from rich.table import Table
from dataclasses import dataclass

from ecoml.data_preparation.pytorch_utils import read_layers_info
from ecoml.model_builder.model_inference import InferenceModel

console = Console()
error_console = Console(stderr=True, style="bold red")

@dataclass
class InferenceResult:
    name: str
    ltype: str
    runtime: float

def run_inference(model_sumary_path: Path, power_profiles: dict[str, int], verbose: bool = False) -> list[InferenceResult]:
    convolution = InferenceModel(model_version=1, layer_type="convolutional", verbose=verbose)
    pooling = InferenceModel(model_version=1, layer_type="pooling", verbose=verbose)
    dense = InferenceModel(model_version=1, layer_type="dense", verbose=verbose)

    layer_info_read = read_layers_info(model_sumary_path)
    
    if verbose:
        print(f"Found {len(layer_info_read)} layers in {model_sumary_path}.")

    inference_results = []
    for layer_name, layer_info in layer_info_read.items():
        layer_type = layer_info.get_layer_type()

        if layer_type == "convolutional":
            model = convolution
        elif layer_type == "pooling":
            model = pooling
        elif layer_type == "dense":
            model = dense
        else:
            if verbose:
                print(f"Skipping layer: {layer_name}")
            continue

        features = model.get_features(layer_info)
        predicted_runtime = model.runtime_model.predict(features.values).tolist()[0]

        inference_results.append(InferenceResult(layer_name, layer_type, predicted_runtime))

    
    return inference_results


def get_metrics(df: pd.DataFrame, cfg: dict[str, int]) -> pd.DataFrame:
    """Calculate predicted runtime, power and energy metrics.

    Args:
        df: Input dataframe used to calculate metrics
        cfg: Dictionary containing various power profiles

    Returns:
        DataFrame containing predicted energy, runtime and power metrics.
    """
    # Convert runtime from milliseconds to seconds
    df["runtime_prediction"] = df["runtime_prediction"] / 1000
    # Total predicted runtime
    total_runtime = df["runtime_prediction"].sum()

    power_list, energy_list = [], []
    for power_val in cfg.values():
        # Average power
        avg_power_consumed = (
            power_val * df["runtime_prediction"]
        ).sum() / total_runtime
        power_list.append(avg_power_consumed)

        # Total energy consumption
        total_energy = (power_val * df["runtime_prediction"]).sum()
        energy_list.append(total_energy)

    metrics_df = pd.DataFrame(
        {
        # zip(power, [total_runtime] * len(power), energy),
        # columns=["power", "latency", "energy"],
            "power": power_list,
            "latency": [total_runtime] * len(power_list),
            "energy": energy_list,
        }
    )
    return metrics_df


def display_metrics_table(runtime_predictions: list[InferenceResult], power_profiles: dict[str, int]) -> None:
    """Display a table of energy, power and latencies for various power profiles.

    Args:
        metrics_df: DataFrame containing predicted energy, runtime and power metrics.
    """

    total_runtime = sum(r.runtime for r in runtime_predictions) / 1000

    energy = [pwr * total_runtime for pwr in power_profiles.values()]

    energy_stats = {
        "Min": f"{min(energy):.3f}",
        "Avg": f"{mean(energy):.3f}",
        "Max": f"{max(energy):.3f}",
    }

    table = Table(
        title="Energy Consumption",
        show_lines=True,
        caption_justify="left",
    )
    # stats_labels = ["Min", "Avg", "Max"]
    table.add_column("Statistic", justify="center", style="cyan")
    table.add_column("Consumption (J)", justify="center", style="magenta") 

    for label, value in energy_stats.items():
        table.add_row(label, value)

    console.print(table)

def display_runtime_table(runtime_predictions: list[InferenceResult]) -> None:
    runtime_table = Table(
        title="Runtime Table",
        show_lines=True,
        caption="Showing predicted runtime across power levels in milliseconds"
    )
    runtime_table.add_column("Predicted runtime (ms)", justify="center", style="green")

    total_runtime = sum(map(lambda i : i.runtime, runtime_predictions))

    runtime_table.add_row(f"{total_runtime:.3f}")

    console.print(runtime_table)

def display_latency_table(runtime_predictions: list[InferenceResult]) -> None:
    """Display a table of layer name and predicted runtime for the layer.

    Args:
        df:  Input dataframe containing layer-wise latency
    """
    table = Table(title="PyTorch Model Layer-wise Latency")
    table.add_column(
        "Layer Name",
        justify="center",
        style="cyan",
        no_wrap=True,
    )
    table.add_column("Predicted runtime (milliseconds)", justify="center", style="green")
    
    for prediction in runtime_predictions:
        table.add_row(prediction.name, str(prediction.runtime))

    console.print(table)

def display_comparison_table(baseline_results: list[InferenceResult], compare_results: list[InferenceResult], power_profiles: dict[str, int]) -> None:
    if not baseline_results or not compare_results:
        console.print("Cannot compare results")
        return
    
    baseline_runtime = sum(r.runtime for r in baseline_results) 
    compare_runtime = sum(r.runtime for r in compare_results)

    baseline_energy = [pwr * baseline_runtime for pwr in power_profiles.values()]
    compare_energy = [pwr * compare_runtime for pwr in power_profiles.values()]

    baseline_energy_avg = mean(baseline_energy) / 1000
    compare_energy_avg = mean(compare_energy) / 1000

    runtime_improvement = (baseline_runtime - compare_runtime) / baseline_runtime * 100
    energy_improvement = (baseline_energy_avg - compare_energy_avg) / baseline_energy_avg * 100

    runtime_grade = "green" if runtime_improvement > 0 else "red"
    energy_grade = "green" if energy_improvement > 0 else "red"

    table = Table(title="Comparison between models", show_lines=True)
    table.add_column("Metric", justify="left", style="cyan")
    table.add_column("Baseline", justify="right", style="white")
    table.add_column("Improved", justify="right", style="white")
    table.add_column("Difference (%)", justify="right", style="green")

    table.add_row(
        "Total runtime (ms)",
        f"{baseline_runtime:.3f}",
        f"{compare_runtime:.3f}",
        f"[{runtime_grade}]{runtime_improvement:.3f}[/{runtime_grade}]"
    )

    table.add_row(
        "Avg Energy (J)",
        f"{baseline_energy_avg:.3f}",
        f"{compare_energy_avg:.3f}",
        f"[{energy_grade}]{energy_improvement:.3f}[/{energy_grade}]"
    )

    console.print(table)


# Get rid of verbose flag, replace print with logs (later)

# is 5 seconds (resnet) lining up with our data. Same with joules

# "if a --scenario flag, maybe predict the quantised stuff" - possibly