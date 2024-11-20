"""Training pipeline."""

from pathlib import Path
from typing import Any

from config.convolutional_features import CONV_FEATURES, CONVOLUTION_PIPELINE
from config.dense_features import DENSE_FEATURES, DENSE_PIPELINE
from config.pooling_features import POOLING_FEATURES, POOLING_PIPELINE
from data_preparation.io_utils import read_yaml_file
from pipeline.trainer import Trainer


def get_config(config_path: Path = Path("config/config.yaml")) -> Any:
    """Get configuration for training and logging the model.

    Args:
        config_path: Path to configuration file.
            Defaults to Path("config/config.yaml").

    Returns:
        Dictionary containnig configuration.
    """
    return read_yaml_file(config_path)


def train_pipeline(
    layer_type: str,
    model_type: str,
    data_config: dict,
    features: list[str],
    pipeline_parameters: dict[str, Any],
    pattern: str,
) -> None:
    """Training pipeline.

    Args:
        layer_type: Type of layer for which training is being performed.
        model_type: Type of model to be trained.
            It can be either power or runtime.
        data_config: Configuration related to data
        features: List of columns to be used as features.
        pipeline_parameters: Paramters used to construct a sklearn pipeline
        pattern: Pattern used by rglob to find relevant CSV files.
    """
    params = pipeline_parameters[model_type]
    trainer = Trainer(data_config=data_config, features=features)

    dataset = trainer.get_dataset(pattern=pattern)
    if dataset is None:
        print("No dataset found for training")
        return

    pipeline = trainer.get_model(
        features_mapping=trainer.dataset_builder.features_mapping,
        polynomial_degree=params["degree"],
        is_log=params.get("is_log", False),
        special_terms_list=params.get("special_terms_list", None),
    )
    trainer.train_and_eval_pipeline(
        dataset=dataset,
        pipeline=pipeline,
        layer_type=layer_type,
        model_type=model_type,
    )


def main(config: dict) -> None:
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
    for model_type in ["power", "runtime"]:
        print("-" * 80)
        print(f"Training for layer = convolutional and model = {model_type}")
        train_pipeline(
            layer_type="convolutional",
            model_type=model_type,
            data_config=data_config,
            features=CONV_FEATURES,
            pipeline_parameters=CONVOLUTION_PIPELINE,
            pattern="**/convolutional.csv",
        )

    # Train power and runtime for pooling layer
    for model_type in ["power", "runtime"]:
        print("-" * 80)
        print(f"Training for layer = pooling and model = {model_type}")
        train_pipeline(
            layer_type="pooling",
            model_type=model_type,
            data_config=data_config,
            features=POOLING_FEATURES,
            pipeline_parameters=POOLING_PIPELINE,
            pattern="**/pooling.csv",
        )

    # Train power and runtime for pooling layer
    for model_type in ["power", "runtime"]:
        print("-" * 80)
        print(f"Training for layer = dense and model = {model_type}")
        train_pipeline(
            layer_type="dense",
            model_type=model_type,
            data_config=data_config,
            features=DENSE_FEATURES,
            pipeline_parameters=DENSE_PIPELINE,
            pattern="**/dense.csv",
        )


if __name__ == "__main__":
    config = get_config()
    main(config)
