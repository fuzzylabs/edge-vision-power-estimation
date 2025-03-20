"""Load inference model from Mlflow registry."""

import os
from pathlib import Path
from typing import Any, Literal, Optional

import dagshub
import mlflow
import pandas as pd
from rich import print
from loguru import logger

from ecoml.data_preparation.features import (
    get_convolutional_features,
    get_dense_features,
    get_pooling_features,
)
from ecoml.data_preparation.pytorch_utils import PytorchLayer

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
        dagshub_repo_owner: Optional[str] = "fuzzylabs",
        dagshub_repo_name: Optional[str] = "edge-vision-power-estimation",
    ):
        self.layer_type = layer_type
        self.model_version = model_version
        self.verbose = verbose
        self.repo_name = dagshub_repo_name
        self.repo_owner = dagshub_repo_owner
        self.base_model_dir = Path.cwd() / "ecoml_models" / self.layer_type
        # Download model from MLFlow Registry if not present on first run
        self.runtime_model = self.load_model(model_type="runtime")
        

    def _download_model(self, model_uri: str, dst_path: str) -> None:
        """Download model from MLflow registry to local filesystem.

        Args:
            model_uri: URI pointing to model artifact.
            dst_path: Path of the local filesystem destination directory
                to which to download the specified artifacts.
        """
        if self.verbose:
            logger.info(f"Downloading model to {dst_path} folder")
        dagshub.init(repo_name=self.repo_name, repo_owner=self.repo_owner, mlflow=True)
        mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=str(dst_path))

    def load_model(self, model_type: str) -> Any:
        """Download and load power or runtime model from MLflow Registry.

        Download is skipped if model exists in the local filesystem.

        Returns:
            Power or runtime model from MLflow Registry.
        """
        model_name = f"{self.layer_type}_{model_type}_model"
        model_uri = f"models:/{model_name}/{self.model_version}"
        dst_path = self.base_model_dir / model_type
        # TODO: Tighter check to see if current model version is present
        # instead of checking only if directory exists
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
                logger.info("Model already downloaded")

        if self.verbose:
            logger.info(f"Loading the {model_type} trained model from {dst_path} folder")
        logger.info(f"Loading {model_type} model from {dst_path}")

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
        elif self.layer_type == "pooling":
            features = get_pooling_features(layer_info)
        elif self.layer_type == "dense":
            features = get_dense_features(layer_info)
        else:
            raise ValueError(f"Unsupported layer type: {self.layer_type}")
        
        return pd.DataFrame([features])
