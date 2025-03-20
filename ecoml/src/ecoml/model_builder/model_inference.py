"""Load inference model from MLflow registry."""

import os
from pathlib import Path
from typing import Any, Literal
import dagshub
import mlflow
import pandas as pd
from loguru import logger

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

    Downloads model from MLflow Registry on DagsHub if not already present.
    """

    def __init__(
        self,
        layer_type: ALLOWED_LAYER_TYPES,
        model_version: int,
        verbose: bool = False,
        dagshub_repo_owner: str = "fuzzylabs",
        dagshub_repo_name: str = "edge-vision-power-estimation",
        use_packaged_models: bool = False,
    ):
        self.layer_type = layer_type
        self.model_version = model_version
        self.verbose = verbose
        self.repo_name = dagshub_repo_name
        self.repo_owner = dagshub_repo_owner
        self.use_packaged_models = use_packaged_models

        if not self.use_packaged_models:
            self.base_model_dir = Path.cwd() / "ecoml_models" / self.layer_type
        else:
            self.base_model_dir = pkg_resources.files(ecoml).joinpath("ecoml_models", self.layer_type)

        self.runtime_model = self.load_model(model_type="runtime")

    def _download_model(self, model_uri: str, dst_path: Path) -> None:
        """Download model from MLflow registry to local filesystem.

        Args:
            model_uri: URI pointing to model artifact.
            dst_path: Destination directory path.
        """
        if self.verbose:
            logger.info(f"Downloading model to {dst_path}")
        dagshub.init(repo_name=self.repo_name, repo_owner=self.repo_owner, mlflow=True)
        mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=str(dst_path))

    def load_model(self, model_type: str) -> Any:
        """Download and load runtime or power model from MLflow Registry.

        Returns:
            Loaded runtime or power model.
        """
        if self.use_packaged_models:
            model_path = self.base_model_dir / model_type / "model.pkl"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Packaged model file not found: {model_path}. "
                    "Ensure ecoml_models are packaged correctly."
                )
            model_dir = model_path.parent
            if self.verbose:
                logger.info(f"Loading packaged {model_type} model from {model_dir}")
            return mlflow.pyfunc.load_model(str(model_dir))

        # Using DagsHub / MLFlow remote model
        model_name = f"{self.layer_type}_{model_type}_model"
        model_uri = f"models:/{model_name}/{self.model_version}"
        dst_path = self.base_model_dir / model_type
        version_file = dst_path / "version.txt"

        need_download = True
        if version_file.exists():
            local_version = version_file.read_text().strip()
            if local_version == str(self.model_version):
                need_download = False

        if need_download:
            self._download_model(model_uri=model_uri, dst_path=dst_path)
            version_file.write_text(str(self.model_version))
        else:
            if self.verbose:
                logger.info(f"{model_type.capitalize()} model version {self.model_version} already downloaded.")

        if self.verbose:
            logger.info(f"Loading {model_type} model from {dst_path}")

        return mlflow.pyfunc.load_model(str(dst_path))

    def get_features(self, layer_info: PytorchLayer) -> pd.DataFrame:
        """Get features for prediction based on layer type.

        Args:
            layer_info: Layer details.

        Returns:
            DataFrame containing features.
        """
        if self.layer_type == "convolutional":
            features = get_convolutional_features(layer_info)
        elif self.layer_type == "pooling":
            features = get_pooling_features(layer_info)
        elif self.layer_type == "dense":
            features = get_dense_features(layer_info)
        else:
   
