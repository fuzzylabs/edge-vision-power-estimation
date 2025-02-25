"""Run inference for PyTorch model."""

from collections import defaultdict
from pathlib import Path

import pandas as pd
from rich import print
from rich.console import Console
from rich.table import Table

from ecoml.data_preparation.pytorch_utils import read_layers_info
from ecoml.model_builder.model_inference import InferenceModel

console = Console()
error_console = Console(stderr=True, style="bold red")


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


def display_metrics_table(metrics_df: pd.DataFrame) -> None:
    """Display a table of energy, power and latencies for various power profiles.

    Args:
        metrics_df: DataFrame containing predicted energy, runtime and power metrics.
    """
    table = Table(
        title="Energy Consumption",
        show_lines=True,
        caption=(
            "This table shows energy consumption where:\n"
            "- [bold]Min[/bold]: The lowest amount of predicted energy.\n"
            "- [bold]Avg[/bold]: The average amount of energy across measurements.\n"
            "- [bold]Max[/bold]: The maximum amount of predicted energy."
        ),
        caption_justify="left",
    )
    stats_labels = ["Min", "Avg", "Max"]
    table.add_column("Statistic", justify="center", style="cyan")
    table.add_column("Consumption (J)", justify="center", style="magenta")

    # table.add_column("Power (W)", justify="center", style="cyan")
    # table.add_column("Runtime (s)", justify="center", style="green")
    # table.add_column("Energy (J)", justify="center", style="magenta")

    for label, (_, row) in zip(stats_labels, metrics_df.iterrows()):
        table.add_row(
            label,
            f"{row['energy']:.3f}"
        )
    console.print(table)

def display_runtime_table(metrics_df: pd.DataFrame) -> None:
    runtime_table = Table(
        title="Runtime Table",
        show_lines=True,
        caption="Showing predicted runtimes (S)"
    )
    runtime_table.add_column("Predicted runtime (s)", justify="center", style="green")

    for _, runtime in metrics_df.iterrows():
        runtime_table.add_row(f"{runtime['latency']:.3f}")

    console.print(runtime_table)

def display_latency_table(df: pd.DataFrame) -> None:
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
    table.add_column("Predicted runtime (seconds)", justify="center", style="green")
    for _, row in df.iterrows():
        table.add_row(str(row["layer_name"]), str(row["runtime_prediction"]))
    console.print(table)


def run_inference(
    model_summary_path: Path, # Model summary not path
    power_profiles: dict[str, int],
    verbose: bool = False,
) -> None: # Return dict
    """Perform inference for a given PyTorch engine file.

    DagsHub related configuration is used to pull models from
    MLflow Registry. Models are pulled from MLflow registry
    for performing prediction.

    Args:
        model_summary_path: Path to pytorch model summary file.
        power_profiles: Power value for various power profiles
        verbose: Show detailed output logs
    """
    # TODO: Expose model version via cli
    conv_models = InferenceModel(
        model_version=1, layer_type="convolutional", verbose=verbose
    )
    pooling_models = InferenceModel(
        model_version=1, layer_type="pooling", verbose=verbose
    )
    dense_models = InferenceModel(model_version=1, layer_type="dense", verbose=verbose)

    data = defaultdict(list)
    layers_info = read_layers_info(model_summary_path) # Should be layer info not path
    if verbose:
        print(f"Found {len(layers_info)} number of layers")
        print(f"Performing inference for {model_summary_path}")

    for layer_name, layer_info in layers_info.items():
        layer_type = layer_info.get_layer_type()
        if layer_type == "convolutional":
            model = conv_models
        elif layer_type == "pooling":
            model = pooling_models
        elif layer_type == "dense":
            model = dense_models
        else:
            if verbose:
                print(f"Skipping layer: {layer_name}")
            continue

        features = model.get_features(layer_info)
        data["runtime_prediction"].append(
            model.runtime_model.predict(features.values).tolist()[0]
        )
        data["layer_name"].append(layer_name)
        data["layer_type"].append(layer_info.layer_type)

    if not len(data):
        error_console.print(
            "Looks like there are no convolutional, pooling or linear layers in the model"
        )
        return

    data["low_power_prediction"] = [power_profiles["low"]] * len(data["layer_name"])
    data["average_power_prediction"] = [power_profiles["average"]] * len(
        data["layer_name"]
    )
    data["high_power_prediction"] = [power_profiles["high"]] * len(data["layer_name"])

    df = pd.DataFrame.from_dict(data)
    if verbose:
        display_latency_table(df)

    metrics_df = get_metrics(df, cfg=power_profiles)
    display_metrics_table(metrics_df)
    display_runtime_table(metrics_df)


# 1 make table neat
# 2 Refactor function ^
# Get rid of verbose flag, replace print with logs (later)