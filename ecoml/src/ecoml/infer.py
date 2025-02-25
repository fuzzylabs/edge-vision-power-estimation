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

    power, energy = [], []
    for power_val in cfg.values():
        # Average power
        avg_power_consumed = (
            power_val * df["runtime_prediction"]
        ).sum() / total_runtime
        power.append(avg_power_consumed)

        # Total energy consumption
        total_energy = (power_val * df["runtime_prediction"]).sum()
        energy.append(total_energy)

    metrics_df = pd.DataFrame(
        zip(power, [total_runtime] * len(power), energy),
        columns=["power", "latency", "energy"],
    )
    return metrics_df


def display_metrics_table(metrics_df: pd.DataFrame) -> None:
    """
    Display a table of energy, power and latencies for various power profiles.

    Args:
        metrics_df: DataFrame containing predicted energy, runtime and power metrics.
    """
    table = Table(title="PyTorch Model Estimations")

    table.add_column(
        "Average power consumption (Watts)",
        justify="center",
        style="cyan",
        no_wrap=True,
    )
    table.add_column(
        "Predicted runtime (seconds)",
        justify="center",
         style="green"
    )
    table.add_column(
        "Average energy consumption (Joules)", 
        justify="center", 
        style="magenta"
    )

    for _, row in metrics_df.iterrows():
        table.add_row(
            f"{row['power']:.3f}",
            f"{row['latency']:.3f}",
            f"{row['energy']:.3f}"
        )
        
    console.print(table)


def display_latency_table(df: pd.DataFrame) -> None:
    """
    Display a table of layer name and predicted runtime for the layer.

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
    table.add_column(
        "Predicted runtime (seconds)", 
        justify="center", 
        style="green"
    )

    for _, row in df.iterrows():
        table.add_row(
            str(row["layer_name"]), 
            f"{row['runtime_prediction']:.3f}"
        )
    console.print(table)


def run_inference(
    model_summary_path: Path,
    power_profiles: dict[str, int],
    verbose: bool = False,
) -> None:
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
    layers_info = read_layers_info(model_summary_path)
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
