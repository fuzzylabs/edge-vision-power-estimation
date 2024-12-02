"""Inference script.

Usage:
python inference.py \
    --trt-engine-path sample_data/resnet18_trt_engine_info.json \
    --result-csv-path results/resnet18_predictions.csv
"""

import argparse
from pathlib import Path

import dagshub
import pandas as pd
from data_preparation.tensorrt_utils import read_layers_info
from model.model_inference import InferenceModel


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

    conv_models = InferenceModel(model_version=1, model_layer_type="convolutional")
    pooling_models = InferenceModel(model_version=1, model_layer_type="pooling")
    dense_models = InferenceModel(model_version=1, model_layer_type="dense")

    power_pred, runtime_pred, layer_names, layer_types = [], [], [], []
    layers_info = read_layers_info(trt_engine_info_path)
    print(f"Performing inference for {trt_engine_info_path}")
    print(f"Found {len(layers_info)} number of layers")

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
        power_pred.append(model.power_model.predict(features.values).tolist()[0])
        runtime_pred.append(model.runtime_model.predict(features.values).tolist()[0])
        layer_names.append(layer_name)
        layer_types.append(layer_info.layer_type)

    data = {
        "layer_type": layer_types,
        "layer_name": layer_names,
        "power_prediction": power_pred,
        "runtime_prediction": runtime_pred,
    }

    df = pd.DataFrame.from_dict(data)
    result_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not (result_csv_path.suffix == ".csv"):
        raise ValueError(f"{result_csv_path} path to csv must end with .csv")
    df.to_csv(result_csv_path, index=False)


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

    infer(
        dagshub_repo_name=args.name,
        dagshub_repo_owner=args.owner,
        trt_engine_info_path=Path(args.trt_engine_path),
        result_csv_path=Path(args.result_csv_path),
    )
