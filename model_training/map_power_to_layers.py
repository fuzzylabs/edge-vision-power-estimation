"""This script maps the power to the individual layer for each model.

Example command:
    python map_power_to_layers.py \
        --raw-dir raw_data \
        --result-dir results
"""

import argparse
from pathlib import Path

from data_preparation.data_preprocess import DataPreprocessing
from data_preparation.io_utils import get_idle_power_log_file, parse_model_dir


def main(args: argparse.Namespace) -> None:
    """Convert raw dataset to preprocessed dataset.

    Args:
        args: CLI arguments
    """
    idle_power_log_path = get_idle_power_log_file(args.raw_data_dir)
    print(f"Idle power log file: {idle_power_log_path}")

    preprocessor = DataPreprocessing(
        idle_power_log_path=idle_power_log_path, result_dir=args.result_dir
    )

    raw_data_path = Path(args.raw_data_dir)
    model_count = sum(1 for model_dir in raw_data_path.iterdir() if model_dir.is_dir())
    print(f"Found {model_count} models in raw data.")

    # Convert and save each model directory raw data to preprocessed data
    for model_dir in raw_data_path.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name
            print(f"Preprocessing {model_name} model")
            try:
                power_log_file, trt_layer_latency_file, trt_engine_info_file = (
                    parse_model_dir(model_dir)
                )
                metrics_by_cycle = preprocessor.compute_layer_metrics_by_cycle(
                    power_log_path=power_log_file,
                    trt_layer_latency_path=trt_layer_latency_file,
                    trt_engine_info_path=trt_engine_info_file,
                )
                preprocessor.save_result_to_csv(
                    metrics_by_cycle=metrics_by_cycle, model_name=model_name
                )
                preprocessor.copy_trt_engine_to_target_dir(
                    model_name=model_name, trt_engine_info_path=trt_engine_info_file
                )

            except ValueError as e:
                print(f"Skipping {model_dir.name} due to error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Power mapping with logs collected on Jetson",
        description="Map power usage data during inference cycles for CNN models.",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default="raw_data",
        help="Path to raw data directory.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="preprocessed_data",
        help="The directory to save the csv.",
    )
    args = parser.parse_args()

    main(args)
