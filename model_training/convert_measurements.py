"""Script to convert measurements taken on an edge device into training data."""

import argparse
from pathlib import Path

from data_preparation.convert import convert_measurements_to_training_data
from data_preparation.measurement_utils import preprocess_measurement_data
from data_preparation.tensorrt_utils import read_layers_info


def main(args: argparse.Namespace) -> None:
    """Convert measurements taken on an edge device into training data.

    Args:
        args: Arguments passed to CLI.
    """
    data_dir = Path(args.preprocessed_data_dir)
    model_dirs = list(data_dir.iterdir())
    print(f"Found {len(model_dirs)} models in preprocessed data folder.")

    # Convert and save each model directory preprocessed data to training data
    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"Preprocessing {model_name} model")
        engine_info_path = f"{model_dir}/trt_engine_info.json"
        measurements_path = f"{model_dir}/power_runtime_mapping_layerwise.csv"
        save_path = Path(f"{args.result_dir}/{model_name}")

        layers_info = read_layers_info(engine_info_path)
        measurements = preprocess_measurement_data(
            measurements_path, args.per_layer_measurements
        )
        # Create training data for a model only if all layers have
        # sufficient samples of power and runtime measurements
        if measurements is not None:
            convert_measurements_to_training_data(save_path, layers_info, measurements)
            print(f"Saved to {save_path}!")
        else:
            print(
                f"Skipping creating training data for {model_name} model. "
                f"It does not have sufficient samples {args.per_layer_measurements} for all the layers."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Convert measurements taken on an edge device into training data."
    )
    parser.add_argument(
        "--preprocessed-data-dir",
        type=str,
        default="preprocessed_data",
        help="Path to preprocessed data directory.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="training_data",
        help="The directory to save training data",
    )
    parser.add_argument(
        "--per-layer-measurements",
        type=int,
        default=10,
        help="Minimum number of measurements for power and runtime in preprocessed data",
    )
    args = parser.parse_args()

    main(args)
