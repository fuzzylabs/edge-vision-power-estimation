"""Visualizes layer wise latency from profiler tracing."""

import glob
from dataclasses import dataclass
from typing import Any
import json
from rich.console import Console
from rich.table import Table
import argparse
from rich.markup import escape
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os


@dataclass
class LayerInfo:
    layer_name: str
    layer_type: str
    input_dimension: list[int]
    output_dimension: list[int]
    latencies: list[list[float]]


def read_trace_file(file_path: str) -> Any:
    """Parse trace file.

    Args:
        file_path: Path to trace file.

    Returns:
        Extracted data.
    """
    with open(file_path, "r") as fp:
        data = json.load(fp)
    return data


def load_trace_data(profiler_dir: str) -> dict[str, LayerInfo]:
    """Extract latencies for each layer across multiple files.

    Args:
        profiler_dir: Path to profiler and trace data

    Returns:
        Dictionary mapping layer of name and
        corresponding latencies for the layer
    """
    layer_data: dict[str, LayerInfo] = {}

    trace_file_paths = (
        f"{profiler_dir}/*/_run_on_acc_0_engine_engine_exectuion_profile.trace"
    )
    for trace_file in tqdm(glob.glob(trace_file_paths), desc="Processing Trace Files"):
        data = read_trace_file(trace_file)

        for layer in data:
            layer_name = layer["name"]
            duration = round(layer["dur"] / 1e3, 3)
            if layer_name not in layer_data:
                layer_data[layer_name] = LayerInfo(
                    layer_name=layer_name,
                    layer_type="",
                    input_dimension=[],
                    output_dimension=[],
                    latencies=[],
                )
            layer_data[layer_name].latencies.append(duration)
    return layer_data


def load_layer_metadata(
    profiler_dir: str, layer_data: dict[str, LayerInfo]
) -> list[LayerInfo]:
    """Extract layer related information.

    Args:
        profiler_dir: Path to profiler and trace data
        layer_data: Dictionary mapping layer of name and
            corresponding latencies for the layer

    Returns:
        List of all metadata related to layer
    """
    layer_file_paths = f"{profiler_dir}/*/_run_on_acc_0_engine_layer_information.json"
    layer_file_path = glob.glob(layer_file_paths)[0]
    layer_info_data = read_trace_file(layer_file_path)

    for layer in layer_info_data["Layers"]:
        layer_name = layer["Name"]
        if layer_name in layer_data:
            layer_data[layer_name].layer_type = layer["LayerType"]
            layer_data[layer_name].input_dimension = layer["Inputs"][0]["Dimensions"]
            layer_data[layer_name].output_dimension = layer["Outputs"][0]["Dimensions"]
    return list(layer_data.values())


def load_layer_data(profiler_dir: str) -> list[LayerInfo]:
    """Extract layer and it's metadata from profiler data.

    Args:
        profiler_dir: Path to profiler and trace data

    Returns:
        List of all metadata related to layer
    """
    layer_data = load_trace_data(profiler_dir)
    return load_layer_metadata(profiler_dir, layer_data)


def create_dataframe(layer_infos: list[LayerInfo]) -> pd.DataFrame:
    """Create a dataframe consisting of only layers and its latencies.

    Args:
        layer_infos: List of all metadata related to layer

    Returns:
        Dataframe containing layer and latency
    """
    data = {"Layer Name": [], "Latency": []}
    for layer in layer_infos:
        data["Layer Name"].extend([layer.layer_name] * len(layer.latencies))
        data["Latency"].extend(layer.latencies)
    df = pd.DataFrame(data)
    return df


def plot_layer_name_latencies(layer_infos: list[LayerInfo], save_fig_dir: str) -> None:
    """Plot a boxplot for layer-wise latencies and save it to a file.

    Args:
        layer_infos: List of all metadata related to layer
        save_fig_dir: Directory to save figure.
    """
    save_path = os.path.join(save_fig_dir, "layer_latencies_boxplot.png")
    df = create_dataframe(layer_infos)
    plt.figure(figsize=(25, 25))
    sns.boxplot(x="Layer Name", y="Latency", data=df)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Layer Name")
    plt.ylabel("Latency (sec)")
    plt.title("Layer-wise Latencies")
    plt.savefig(save_path, dpi=300)
    print(f"Saved boxplot to {save_path}")

    plt.show()


def layer_information(model_name: str, layer_infos: list[LayerInfo]) -> Table:
    """Create a rich table to display layer related information.

    Args:
        model_name: Name of the model
        layer_infos: List of all metadata related to layer

    Returns:
        Table containing layer name, type, input and output dimension
        and average layer-wise latency
    """
    table = Table(title=f"Layer wise Latency for {model_name} model", show_lines=True)
    table.add_column("Layer Name", justify="left", style="magenta")
    table.add_column("Layer Type", style="red")
    table.add_column("Input Dimension", style="cyan")
    table.add_column("Output Dimension", style="blue")
    table.add_column("Average Latency (sec)", justify="right", style="green")

    for layer in layer_infos:
        table.add_row(
            escape(layer.layer_name),
            layer.layer_type,
            str(layer.input_dimension),
            str(layer.output_dimension),
            str(round(np.mean(layer.latencies), 3)),
        )
    return table


def main(args: argparse.Namespace) -> None:
    """Main entrypoint.

    Args:
        args: Arguments from CLI
    """
    console = Console()
    model_name = str(args.profiler_dir.split("/")[1]).capitalize()
    save_fig_dir = "/".join(args.profiler_dir.split("/")[:2])

    # Show layer information and latency
    layer_infos = load_layer_data(args.profiler_dir)
    table = layer_information(model_name, layer_infos)
    console.print(table)

    plot_layer_name_latencies(layer_infos, save_fig_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Visualize tensorrt profiler trace",
    )
    parser.add_argument(
        "--profiler-dir",
        type=str,
        default="results/alexnet/profiling_20241021-160746",
        help="Specify name of profiler folder for the model",
    )

    args = parser.parse_args()
    main(args)
