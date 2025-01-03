"""Training pipeline."""

from pathlib import Path
from typing import Any

from git import Repo
from loguru import logger

from config.convolutional_features import CONV_FEATURES, CONVOLUTION_PIPELINE
from config.dense_features import DENSE_FEATURES, DENSE_PIPELINE
from config.pooling_features import POOLING_FEATURES, POOLING_PIPELINE
from data_preparation.io_utils import read_yaml_file
from trainer.trainer import Trainer


def get_config(config_path: Path = Path("config/config.yaml")) -> Any:
    """Get configuration for training and logging the model.

    Args:
        config_path: Path to configuration file.
            Defaults to Path("config/config.yaml").

    Returns:
        Dictionary containnig configuration.
    """
    return read_yaml_file(config_path)


def get_train_data_version(root_git_dir: str = "..") -> str | None:
    """Get git tag corresponding to training_data.dvc file.

    Args:
        root_git_dir: Path to root of the git repo

    Returns:
        Git tag corresponding to training data

    Raises:
        Exception is thrown if tag is not found for training_data.dvc file
    """
    repo = Repo(root_git_dir)
    current_commit, _ = repo.blame("HEAD", file="model_training/training_data.dvc")[0]
    logger.debug(
        f"Commit corresponding to 'model_training/training_data.dvc' file: {current_commit}"
    )
    git_cmd = repo.git
    try:
        current_tag = git_cmd.tag(current_commit, contains=True)
        logger.debug(
            f"Tag corresponding to 'model_training/training_data.dvc' file: {current_tag}"
        )
        return current_tag
    except Exception as e:
        logger.warning(
            "No tag found for current 'model_training/training_data.dvc' file\n"
            f"Command failed with following error : {e}"
        )
        return None


def train_pipeline(
    layer_type: str,
    model_type: str,
    config: dict,
    features: list[str],
    pipeline_parameters: dict[str, Any],
    pattern: str,
    data_tag: str,
) -> None:
    """Training pipeline.

    Args:
        layer_type: Type of layer for which training is being performed.
        model_type: Type of model to be trained.
            It can be either power or runtime.
        config: Configuration related to data and model parameters
        features: List of columns to be used as features.
        pipeline_parameters: Paramters used to construct a sklearn pipeline
        pattern: Pattern used by rglob to find relevant CSV files.
        data_tag : Name of train data tag used for training.
    """
    params = pipeline_parameters[model_type]
    trainer = Trainer(config=config, features=features)

    dataset = trainer.get_dataset(pattern=pattern)
    if dataset is None:
        logger.warning("No dataset found for training")
        return

    pipeline = trainer.get_model(
        features_mapping=trainer.dataset_builder.features_mapping,
        polynomial_degree=params["degree"],
        is_log=params.get("is_log", False),
        special_terms_list=params.get("special_terms_list", None),
        scaler=params["scaler"],
        lasso_params=params["lasso_params"],
    )
    trainer.train_and_eval_pipeline(
        dataset=dataset,
        pipeline=pipeline,
        layer_type=layer_type,
        model_type=model_type,
        train_data_tag=data_tag,
    )


def main(config: dict) -> None:
    """Training pipeline for all layers.

    Args:
        config: Configuration dict.
    """
    data_tag = get_train_data_version(root_git_dir="..")
    if data_tag is None:
        logger.critical("No data tag found for the training data")
        exit(1)

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
        logger.info(f"Found training data tag: {data_tag}")

        mlflow.sklearn.autolog()

    # Train power and runtime for convolutional layer
    for model_type in ["power", "runtime"]:
        logger.info("-" * 80)
        logger.info(f"Training for layer = convolutional and model = {model_type}")
        train_pipeline(
            layer_type="convolutional",
            model_type=model_type,
            config=config,
            features=CONV_FEATURES,
            pipeline_parameters=CONVOLUTION_PIPELINE,
            pattern="**/convolutional.csv",
            data_tag=data_tag,
        )

    # Train power and runtime for pooling layer
    for model_type in ["power", "runtime"]:
        logger.info("-" * 80)
        logger.info(f"Training for layer = pooling and model = {model_type}")
        train_pipeline(
            layer_type="pooling",
            model_type=model_type,
            config=config,
            features=POOLING_FEATURES,
            pipeline_parameters=POOLING_PIPELINE,
            pattern="**/pooling.csv",
            data_tag=data_tag,
        )

    # Train power and runtime for pooling layer
    for model_type in ["power", "runtime"]:
        logger.info("-" * 80)
        logger.info(f"Training for layer = dense and model = {model_type}")
        train_pipeline(
            layer_type="dense",
            model_type=model_type,
            config=config,
            features=DENSE_FEATURES,
            pipeline_parameters=DENSE_PIPELINE,
            pattern="**/dense.csv",
            data_tag=data_tag,
        )


if __name__ == "__main__":
    config = get_config()
    main(config)
