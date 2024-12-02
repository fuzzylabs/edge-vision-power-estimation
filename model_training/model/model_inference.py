"""Load inference model from Mlflow registry."""

from typing import Any

import mlflow
import pandas as pd

from data_preparation.convert import (
    get_convolutional_features,
    get_dense_features,
    get_pooling_features,
)
from data_preparation.tensorrt_utils import TensorRTLayer


class InferenceModel:
    """Inference Model."""

    def __init__(self, model_layer_type: str, model_version: int):
        self.model_layer_type = model_layer_type
        self.model_version = model_version
        self.power_model = self.load_power_model()
        self.runtime_model = self.load_runtime_model()

    def load_power_model(self) -> Any:
        """Download and load power model from MLflow Registry.

        Returns:
            Power model from MLflow Registry.
        """
        model_name = f"{self.model_layer_type}_power_model"
        return mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{self.model_version}"
        )

    def load_runtime_model(self) -> Any:
        """Download and load runtime model from MLflow Registry.

        Returns:
            Runtime model from MLflow Registry.
        """
        model_name = f"{self.model_layer_type}_runtime_model"
        return mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{self.model_version}"
        )

    def get_features(self, layer_info: TensorRTLayer) -> pd.DataFrame:
        """Get features for the model to run prediction.

        Each layer type 

        Args:
            layer_info: Pydantic class containing all layer information.

        Returns:
            Pandas dataframe containing input features.
        """
        if self.model_layer_type == "convolutional":
            features = get_convolutional_features(layer_info)
        if self.model_layer_type == "pooling":
            features = get_pooling_features(layer_info)
        if self.model_layer_type == "dense":
            features = get_dense_features(layer_info)
        return pd.DataFrame.from_dict([features])
