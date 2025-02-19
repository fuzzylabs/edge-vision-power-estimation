"""Trainer class."""

import inspect
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from dataset_builder.dataset_builder import DatasetBuilder, TrainTestDataset
from loguru import logger
from mlflow.models.signature import infer_signature
from model_builder.model_builder import ModelBuilder
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline


def get_git_branch():
    """Get current branch."""
    return (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode("utf-8")
        .replace("\n", "")
    )


class Trainer:
    def __init__(self, config: dict, features: list[str]) -> None:
        self.data_config = config["data"]
        self.model_config = config["model"]
        self.mlflow_config = config["mlflow"]
        self.features = features
        self.dataset_builder = DatasetBuilder(features=features)
        self.model_builder = ModelBuilder(cv=self.model_config["cross_validation"])

    def get_dataset(self, pattern: str) -> TrainTestDataset:
        """Get train and test dataset.

        Args:
            pattern: Pattern to find relevant CSV data files.

        Returns:
            TrainTestDataset dataclass that contains
            training and testing datasets.
        """
        return self.dataset_builder.create_dataset(
            data_dir=Path(self.data_config["data_dir"]),
            test_models=self.data_config["test_models"],
            pattern=pattern,
        )

    def get_model(
        self,
        features_mapping: dict[str, int],
        polynomial_degree: int,
        scaler: str,
        is_log: bool = False,
        special_terms_list: list[list[str]] | None = None,
        lasso_params: dict[str, Any] = {},
    ) -> Pipeline:
        """Get model to be trained.

        Args:
            features_mapping: Mapping of feature names to indices.
            polynomial_degree: Polynomial degree of regular polynomial terms.
            scaler: Name of sklearn preprocessing scaler
            is_log: Whether to log1p input features.
            special_terms_list: Definitions of special polynomial terms.
            lasso_params: Parameters added to LassoCV sklearn model

        Returns:
            Sklearn pipeline
        """
        return self.model_builder.create_pipeline(
            features_mapping=features_mapping,
            polynomial_degree=polynomial_degree,
            is_log=is_log,
            special_terms_list=special_terms_list,
            scaler=scaler,
            lasso_params=lasso_params,
        )

    def train_and_eval_pipeline(
        self,
        dataset: TrainTestDataset,
        pipeline: Pipeline,
        layer_type: str,
        model_type: str,
        train_data_tag: str,
    ) -> None:
        """Train and evaluation sklearn pipeline.

        If mlflow tracking is enable, this function logs paramters,
        metrics and prediction image to MLFlow.

        Args:
            dataset: Train and test dataset
            pipeline: Sklearn pipeline to be trained.
            layer_type: Type of layer
            model_type: Type of model to be trained.
                It can be either runtime or power.
            train_data_tag : Name of train data tag used for training.
        """
        train_dataset, test_dataset = dataset.train, dataset.test
        logger.info(
            f"Number of CNN models used for training: {len(dataset.train.csv_paths)}"
        )
        logger.info(
            f"Number of CNN models used for testing: {len(dataset.test.csv_paths)}"
        )
        logger.info(f"Training samples: {len(train_dataset.input_features)}")
        logger.info(f"Testing samples: {len(test_dataset.input_features)}")

        train_features = train_dataset.input_features
        test_features = test_dataset.input_features

        if model_type == "power":
            train_target = train_dataset.power
            test_target = test_dataset.power

        if model_type == "runtime":
            train_target = train_dataset.runtime
            test_target = test_dataset.runtime

        # Create mlflow dataset for logging
        train_df = pd.concat([train_features, train_target], axis=1)
        train_mlflow_data = mlflow.data.from_pandas(train_df, targets=model_type)

        test_df = pd.concat([test_features, test_target], axis=1)
        test_mlflow_data = mlflow.data.from_pandas(test_df, targets=model_type)

        logger.info(f"Training {model_type} model")
        mlflow.set_experiment(f"{layer_type}_{model_type}_model")
        mlflow.sklearn.autolog(log_datasets=False, log_models=False)
        with mlflow.start_run(run_name=self.mlflow_config["mlflow_experiment_name"]):
            # MLflow tags
            repo = f"git@github.com:{self.mlflow_config['dagshub_repo_owner']}/{self.mlflow_config['dagshub_repo_name']}.git"
            mlflow.set_tags(
                {
                    "mlflow.source.git.branch": get_git_branch(),
                    "mlflow.source.git.repoURL": repo,
                    "train_data_tag": train_data_tag,
                }
            )

            # Log datasets
            mlflow.log_input(train_mlflow_data, context="Train")
            mlflow.log_input(test_mlflow_data, context="Eval")

            # Train model
            pipeline.fit(train_features.values, train_target.values)

            logger.info(pipeline)
            alpha = pipeline.named_steps["lasso"].alpha_
            coef = pipeline.named_steps["lasso"].coef_
            intercept = pipeline.named_steps["lasso"].intercept_
            n_features_in = pipeline.named_steps["lasso"].n_features_in_
            logger.info(
                f"Lasso model parameters:\nalpha={alpha}\n"
                f"coef={coef}\n"
                f"intercept={intercept}\n"
                f"n_features_in={n_features_in}"
            )

            train_pred = pipeline.predict(train_features.values)
            train_rmspe = Trainer.rmspe_metric(
                actual=train_target.values, pred=train_pred
            )
            logger.info(f"Training RMSPE: {train_rmspe}")
            mlflow.log_metrics(
                {"training_root_mean_squared_percentage_error": train_rmspe}
            )

            # Evaluation
            predictions = pipeline.predict(test_features.values)
            test_metrics = Trainer.eval_metrics(actual=test_target, pred=predictions)
            logger.info(test_metrics)
            mlflow.log_metrics(test_metrics)
            mlflow.log_params(
                {
                    "train_num_cnn_models": len(dataset.train.csv_paths),
                    "test_num_cnn_models": len(dataset.test.csv_paths),
                }
            )

            # Plot and log prediction on mlflow
            test_paths = dataset.test.csv_paths
            for test_file_path in test_paths:
                model_name = test_file_path.parent.stem
                fig = self.plot_layerwise_predictions(
                    test_file_path=test_file_path,
                    pipeline=pipeline,
                    model_type=model_type,
                )
                fig.tight_layout()
                mlflow.log_figure(
                    fig, f"{model_name}_{layer_type}_{model_type}_prediction.png"
                )

            code_path = ["model_builder"]
            signature = infer_signature(
                train_features.values, pipeline.predict(train_features.values)
            )
            mlflow.sklearn.log_model(
                pipeline,
                "model",
                code_paths=code_path,
                signature=signature,
                input_example=train_features.iloc[[0]],
            )

    @staticmethod
    def rmspe_metric(actual, pred) -> float:
        """Calculate root mean squared percentage error metric.

        Args:
            actual: Actual values
            pred: Predicted values

        Returns:
            RMSPE metric.
        """
        EPSILON = 1e-10
        rmspe = np.sqrt(np.mean(np.square((actual - pred) / (actual + EPSILON)))) * 100
        return rmspe

    @staticmethod
    def eval_metrics(actual, pred, prefix: str = "testing_") -> dict[str, float]:
        """Calculate evaluation metrics.

        Args:
            actual: Actual values
            pred: Predicted values
            prefix: Prefix to be added. Default to "testing_".

        Returns:
            Dictionary mapping metric name to it's score.
        """
        rmspe = Trainer.rmspe_metric(actual=actual, pred=pred)
        rmse = root_mean_squared_error(actual, pred)
        mse = mean_squared_error(actual, pred)
        r2 = r2_score(actual, pred)
        mae = mean_absolute_error(actual, pred)
        mape = mean_absolute_percentage_error(actual, pred)
        return {
            f"{prefix}root_mean_squared_percentage_error": rmspe,
            f"{prefix}root_mean_squared_error": rmse,
            f"{prefix}mean_squared_error": mse,
            f"{prefix}r2_score": r2,
            f"{prefix}mean_absolute_error": mae,
            f"{prefix}mean_absolute_percentage_error": mape,
        }

    def plot_layerwise_predictions(
        self, test_file_path: Path, pipeline: Pipeline, model_type: str
    ) -> plt.figure:
        """Plot layerwise prediction for given model and test dataset.

        Args:
            test_file_path: Path to test CSV file.
            features: List of feature column names.
            pipeline: Trained sklearn model
            model_type: Type of trained model.

        Returns:
            Matplotlib figure.
        """
        test_df = self.dataset_builder.read_csv_and_convert_power(
            file_path=test_file_path
        )
        pred = pipeline.predict(test_df[self.features].values)
        test_df[f"{model_type}_pred"] = pred
        test_df = test_df[["layer_name", f"{model_type}", f"{model_type}_pred"]]
        logger.info(
            f"Predictions for {test_file_path.parent.stem} model using {model_type}\n{test_df}"
        )
        # Get first 15 characters from long PyTorch layer names
        test_df.loc[:, "layer_name"] = test_df.loc[:, "layer_name"].str[:15]
        ax = test_df.plot(rot=90, x="layer_name", kind="bar")
        return ax.get_figure()
