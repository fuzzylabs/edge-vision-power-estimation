"""Load inference model from Mlflow registry."""

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

    def load_model(self, model_type: str) -> Any:
        """Download and load power or runtime model from MLflow Registry.

        Returns:
            Power or runtime model from MLflow Registry.
        """
        model_name = f"{self.layer_type}_{model_type}_model"
        return mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{self.model_version}"
        )

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
