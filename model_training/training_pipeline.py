"""Training pipeline."""

from pathlib import Path
from typing import Any

from data_preparation.io_utils import read_yaml_file
from model.convolutional_layer import convolution_pipeline
from model.dense_layer import dense_pipeline
from model.pooling_layer import pooling_pipeline


def get_config(config_path: Path = Path("config/config.yaml")) -> Any:
    """Get configuration for training and logging the model.

    Args:
        config_path: Path to configuration file.
            Defaults to Path("config/config.yaml").

    Returns:
        Dictionary containnig configuration.
    """
    return read_yaml_file(config_path)


def training_pipeline(config: dict) -> None:
    """Training pipeline for all layers.

    Args:
        config: Configuration dict.
    """
    data_config = config["data"]
    mlflow_config = config["mlflow"]

    # Optionally enable mlflow tracking
    if mlflow_config["enable_tracking"]:
        import dagshub
        import mlflow

        dagshub.init(
            repo_name=mlflow_config["dagshub_repo_name"],
            repo_owner=mlflow_config["dagshub_repo_owner"],
            mlflow=True,
        )

        mlflow.set_experiment(mlflow_config["mlflow_experiment_name"])
        mlflow.sklearn.autolog()

    # Train power and runtime for convolutional layer
    convolution_pipeline(data_config)

    # Train power and runtime for pooling layer
    pooling_pipeline(data_config)

    # Train power and runtime for dense layer
    dense_pipeline(data_config)


if __name__ == "__main__":
    config = get_config()
    training_pipeline(config)
