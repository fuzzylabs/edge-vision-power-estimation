"""Inference script.

Usage:
python inference.py \
    --trt-engine-path sample_data/resnet18_trt_engine_info.json \
    --result-csv-path results/resnet18_predictions.csv
"""

import argparse
from collections import defaultdict
from pathlib import Path

import dagshub
import pandas as pd
from data_preparation.tensorrt_utils import read_layers_info
from model.model_inference import InferenceModel


def print_metrics(df: pd.DataFrame) -> None:
    """Print runtime, power and energy metrics.

    Args:
        df: Input dataframe used to calculate metrics
    """
    # Convert runtime from milliseconds to seconds
    df["runtime_prediction"] = df["runtime_prediction"] / 1000

    # Total predicted runtime
    total_runtime = df["runtime_prediction"].sum()

    # Average power
    avg_power_consumed = (
        df["power_prediction"] * df["runtime_prediction"]
    ).sum() / total_runtime

    # Total energy consumption
    total_energy = (df["power_prediction"] * df["runtime_prediction"]).sum()

    print(f"Total runtime : {total_runtime} seconds")
    print(f"Average power consumed : {avg_power_consumed} watts")
    print(f"Total energy spent: {total_energy} joules")


def infer(
    dagshub_repo_owner: str,
    dagshub_repo_name: str,
    trt_engine_info_path: Path,
    result_csv_path: Path,
) -> None:
    """Perform inference for a given TensorRT engine file.

    DagsHub related configuration is used to pull models from
    MLflow Registry. Models are pulled from MLflow registry
    for performing prediction.

    Args:
        dagshub_repo_owner: DagsHub repo
        dagshub_repo_name: DagsHub repo owner
        trt_engine_info_path: Path to tensorrt engine file.
        result_csv_path: Path to save power and runtime prediction

    Raises:
        ValueError: If `result_csv_path` does not end with `.csv`
    """
    dagshub.init(
        repo_name=dagshub_repo_name,
        repo_owner=dagshub_repo_owner,
        mlflow=True,
    )

    conv_models = InferenceModel(model_version=1, layer_type="convolutional")
    pooling_models = InferenceModel(model_version=1, layer_type="pooling")
    dense_models = InferenceModel(model_version=1, layer_type="dense")

    data = defaultdict(list)
    layers_info = read_layers_info(trt_engine_info_path)
    print(f"Found {len(layers_info)} number of layers")
    print(f"Performing inference for {trt_engine_info_path}")

    for layer_name, layer_info in layers_info.items():
        layer_type = layer_info.get_layer_type()
        if layer_type == "convolutional":
            model = conv_models
        elif layer_type == "pooling":
            model = pooling_models
        elif layer_type == "dense":
            model = dense_models
        else:
            continue

        features = model.get_features(layer_info)
        data["power_prediction"].append(
            model.power_model.predict(features.values).tolist()[0]
        )
        data["runtime_prediction"].append(
            model.runtime_model.predict(features.values).tolist()[0]
        )
        data["layer_name"].append(layer_name)
        data["layer_type"].append(layer_info.layer_type)

    df = pd.DataFrame.from_dict(data)
    result_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if result_csv_path.suffix != ".csv":
        raise ValueError(f"{result_csv_path} path to csv must end with .csv")
    df.to_csv(result_csv_path, index=False)
    print_metrics(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference for tensorrt engine info file.")
    parser.add_argument(
        "--owner",
        type=str,
        default="fuzzylabs",
        help="Name of user/organization on DagsHub.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="edge-vision-power-estimation",
        help="The directory to save the log result.",
    )
    parser.add_argument(
        "--trt-engine-path",
        type=str,
        help="Path to tensorrt engine information file.",
    )
    parser.add_argument(
        "--result-csv-path",
        type=str,
        help="Path to save prediction results as a CSV.",
    )
    args = parser.parse_args()

    if not args.trt_engine_path or not args.result_csv_path:
        raise ValueError(
            "Both the flags (--trt-engine-path and --result-csv-path) should be provided"
        )

    infer(
        dagshub_repo_name=args.name,
        dagshub_repo_owner=args.owner,
        trt_engine_info_path=Path(args.trt_engine_path),
        result_csv_path=Path(args.result_csv_path),
    )
