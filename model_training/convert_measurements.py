"""Script to convert measurements taken on an edge device into training data."""

import argparse
from pathlib import Path

from data_preparation.convert import convert_measurements_to_training_data
from data_preparation.measurement_utils import preprocess_measurement_data
from data_preparation.tensorrt_utils import read_layers_info


def main(save_path: Path, measurements_path: Path, engine_info_path: Path) -> None:
    """Convert measurements taken on an edge device into training data.

    Args:
        save_path: Path to save training data
        measurements_path: Path to read power and runtime csv
        engine_info_path: Path to read TensorRT model engine info
    """
    layers_info = read_layers_info(engine_info_path)
    measurements = preprocess_measurement_data(measurements_path)

    convert_measurements_to_training_data(save_path, layers_info, measurements)
    print(f"Saved to {save_path}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Convert measurements taken on an edge device into training data."
    )
    parser.add_argument(
        "save_path", type=Path, help="Path to save the resulting training data."
    )
    parser.add_argument(
        "measurements_path", type=Path, help="Path to the measurements file."
    )
    parser.add_argument(
        "engine_info_path",
        type=Path,
        help="Path to TensorRT engine info generated file.",
    )
    args = parser.parse_args()

    main(args.save_path, args.measurements_path, args.engine_info_path)
