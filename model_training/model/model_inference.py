"""Load inference model from Mlflow registry."""

import os
from typing import Any, Literal

import mlflow
import pandas as pd
from data_preparation.convert import (
    get_convolutional_features,
    get_dense_features,
    get_pooling_features,
)
from data_preparation.tensorrt_utils import TensorRTLayer

ALLOWED_LAYER_TYPES = Literal["convolutional", "pooling", "dense"]


class InferenceModel:
    """Inference Model."""

    def __init__(self, layer_type: ALLOWED_LAYER_TYPES, model_version: int):
        self.layer_type = layer_type
        self.model_version = model_version
        self.power_model = self.load_model(model_type="power")
        self.runtime_model = self.load_model(model_type="runtime")

    def _download_model(self, model_uri: str, dst_path: str) -> None:
        """Download model from MLflow registry to local filesystem.

        Args:
            model_uri: URI pointing to model artifact.
            dst_path: Path of the local filesystem destination directory
                to which to download the specified artifacts.
        """
        print(f"Downloading model to {dst_path} folder")
        mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=dst_path)

    def load_model(self, model_type: str) -> Any:
        """Download and load power or runtime model from MLflow Registry.

        Download is skipped if model exists in the local filesystem.

        Returns:
            Power or runtime model from MLflow Registry.
        """
        model_name = f"{self.layer_type}_{model_type}_model"
        model_uri = f"models:/{model_name}/{self.model_version}"
        dst_path = f"trained_models/{self.layer_type}/{model_type}"
        if not os.path.exists(dst_path):
            self._download_model(model_uri=model_uri, dst_path=dst_path)

        print(f"Loading the {model_type} trained model from {dst_path} folder")
        return mlflow.pyfunc.load_model(dst_path)

    def get_features(self, layer_info: TensorRTLayer) -> pd.DataFrame:
        """Get features for the model to run prediction.

        Each layer type creates input features required by
        power and runtime models using Tensorrt engine info file.

        Args:
            layer_info: Pydantic class containing all layer information.

        Returns:
            Pandas dataframe containing input features.
        """
        if self.layer_type == "convolutional":
            features = get_convolutional_features(layer_info)
        if self.layer_type == "pooling":
            features = get_pooling_features(layer_info)
        if self.layer_type == "dense":
            features = get_dense_features(layer_info)
        return pd.DataFrame.from_dict([features])
