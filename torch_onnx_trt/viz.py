"""Visualize latency and throughput across all backends."""

from typing import Any
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def read_json_file(file_path: str) -> Any:
    """Parse JSON file.

    Args:
        file_path: Path to JSON file.

    Returns:
        Extracted data.
    """
    with open(file_path, "r") as fp:
        data = json.load(fp)
    return json.loads(data)


def create_dataframe(json_files: list[str]) -> pd.DataFrame:
    """Create dataframe to collect data from all json files.

    Args:
        json_files: List of path to json files

    Returns:
        DataFrame containing data from all json files.
    """
    data = {"config": [], "backend": [], "avg_throughput": [], "avg_latency": []}
    for file in json_files:
        file_data = read_json_file(file)
        data["config"].append(file_data["config"])
        data["backend"].append(file_data["config"]["backend"])
        data["avg_throughput"].append(file_data["avg_throughput"])
        data["avg_latency"].append(file_data["avg_latency"])
    return pd.DataFrame(data)


def plot_latency_throughput(df: pd.DataFrame, save_dir: str) -> None:
    """Plot latency and throughput across all backends.

    Args:
        df: DataFrame containing data from all json files.
        save_dir: Directory to save the figure.
    """
    save_path = Path(save_dir) / "latency_throughput.png"
    config = df["config"].iloc[0]

    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.bar(df["backend"], df["avg_throughput"], color=["blue", "green", "red"])
    plt.title("Average Throughput")
    plt.xlabel("Backend")
    plt.ylabel("Throughput (img/sec)")
    plt.grid(axis="y")

    plt.subplot(1, 2, 2)
    plt.bar(df["backend"], df["avg_latency"], color=["blue", "green", "red"])
    plt.title("Average Latency")
    plt.xlabel("Backend")
    plt.ylabel("Latency (seconds)")
    plt.grid(axis="y")

    plt.figtext(
        0.5,
        0.02,
        f"Config: dtype={config['dtype']}",
        ha="center",
        fontsize=15,
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved boxplot to {save_path}")
    plt.show()


def main(args: argparse.Namespace) -> None:
    """Main entrypoing

    Args:
        args: Arguments from CLI
    """
    # Get last 3 benchmark files
    # TODO: Limitation when multiple benchmark json files are present.
    json_files = list(Path(args.model_dir).glob("*.json"))[:3]
    df = create_dataframe(json_files)
    plot_latency_throughput(df, args.model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Visualize latency and througput for all backends.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="results/alexnet",
        help="Specify the folder where results for the model are stored.",
    )

    args = parser.parse_args()
    main(args)
