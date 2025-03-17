"""Load inference model from Mlflow registry."""

import os
from typing import Any, Literal, Optional

import dagshub
import mlflow
import pandas as pd
from rich import print

from ecoml.data_preparation.features import (
    get_convolutional_features,
    get_dense_features,
    get_pooling_features,
)
from ecoml.data_preparation.pytorch_utils import PytorchLayer
import importlib.resources as pkg_resources
import ecoml

ALLOWED_LAYER_TYPES = Literal["convolutional", "pooling", "dense"]


class InferenceModel:
    """Inference Model.

    It downloads model from MLFlow Registry on DagsHub, if not present on the first run.
    """

    def __init__(
        self,
        layer_type: ALLOWED_LAYER_TYPES,
        model_version: int,
        verbose: bool = False,
    ):
        self.layer_type = layer_type
        self.model_version = model_version
        self.verbose = verbose
        # Download model from MLFlow Registry if not present on first run
        self.runtime_model = self.load_model(model_type="runtime")
        
    def load_model(self, model_type: str) -> Any:
        """Download and load power or runtime model from MLflow Registry.

        Download is skipped if model exists in the local filesystem.

        Returns:
            Power or runtime model from MLflow Registry.
        """
        base_path = pkg_resources.files(ecoml).joinpath("ecoml_models")
        model_path = base_path.joinpath(f"{self.layer_type}/{model_type}/model.pkl")

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}. Ensure ecoml_models is packaged correctly.")

        model_dir = model_path.parent

        if self.verbose:
            print(f"Loading the {model_type} trained model from {model_dir}")

        return mlflow.pyfunc.load_model(str(model_dir))

    def get_features(self, layer_info: PytorchLayer) -> pd.DataFrame:
        """Get features for the model to run prediction.

        Each layer type creates input features required by
        power and runtime models using Pytorch model summary file.

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
