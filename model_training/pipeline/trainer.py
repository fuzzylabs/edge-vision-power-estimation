"""Trainer class."""

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from dataset.dataset_builder import DatasetBuilder, TrainTestDataset
from model.model_builder import ModelBuilder
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline


class Trainer:
    def __init__(self, data_config: dict, features: list[str]) -> None:
        self.data_config = data_config
        self.features = features
        self.dataset_builder = DatasetBuilder(features=features)
        self.model_builder = ModelBuilder(cv=self.data_config["cross_validation"])

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
        is_log: bool = False,
        special_terms_list: list[list[str]] | None = None,
    ) -> Pipeline:
        """Get model to be trained.

        Args:
            features_mapping (dict[str, int]): Mapping of feature names to indices.
            polynomial_degree (int): Polynomial degree of regular polynomial terms.
            is_log (bool): Whether to log1p input features.
            special_terms_list (list[list[str]]): Definitions of special polynomial terms.

        Returns:
            Sklearn pipeline
        """
        return self.model_builder.create_pipeline(
            features_mapping=features_mapping,
            polynomial_degree=polynomial_degree,
            is_log=is_log,
            special_terms_list=special_terms_list,
        )

    def train_and_eval_pipeline(
        self,
        dataset: TrainTestDataset,
        pipeline: Pipeline,
        layer_type: str,
        model_type: str,
    ) -> None:
        """Train and evaluation sklearn pipeline.

        If mlflow tracking is enable, this function logs paramters,
        metrics and prediction image to MLFlow.

        Args:
            dataset: Train and test dataset
            pipeline: Sklearn pipeline to be trained.
            layer_type: T
            model_type: Type of model to be trained.
                It can be either runtime or power.
        """
        train_dataset, test_dataset = dataset.train, dataset.test
        print(f"Number of CNN models used for training: {len(dataset.train.csv_paths)}")
        print(f"Number of CNN models used for testing: {len(dataset.test.csv_paths)}")
        print(f"Training samples: {len(train_dataset.input_features)}")
        print(f"Testing samples: {len(test_dataset.input_features)}")

        train_features = train_dataset.input_features.values
        test_features = test_dataset.input_features.values
        if model_type == "power":
            train_target = train_dataset.power.values
            test_target = test_dataset.power
        if model_type == "runtime":
            train_target = train_dataset.runtime.values
            test_target = test_dataset.runtime

        print(f"Training {model_type} model")
        with mlflow.start_run(run_name=f"{layer_type}_{model_type}_model"):
            # Train model
            pipeline.fit(train_features, train_target)
            print(pipeline)
            print(
                pipeline.named_steps["lasso"].alpha_,
                pipeline.named_steps["lasso"].coef_,
                pipeline.named_steps["lasso"].intercept_,
                pipeline.named_steps["lasso"].n_features_in_,
            )
            train_pred = pipeline.predict(train_features)
            train_rmspe = self.rmspe_metric(actual=train_target, pred=train_pred)
            mlflow.log_metrics(
                {"training_root_mean_squared_percentage_error": train_rmspe}
            )

            # Evaluation
            predictions = pipeline.predict(test_features)
            test_metrics = self.eval_metrics(actual=test_target, pred=predictions)
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

    def rmspe_metric(self, actual, pred) -> float:
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

    def eval_metrics(self, actual, pred, prefix: str = "testing_") -> dict[str, float]:
        """Calculate evaluation metrics.

        Args:
            actual: Actual values
            pred: Predicted values
            prefix: Prefix to be added. Default to "testing_".

        Returns:
            Dictionary mapping metric name to it's score.
        """
        rmspe = self.rmspe_metric(actual=actual, pred=pred)
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
        print(f"Predictions for {test_file_path.parent.stem} model using {model_type}")
        print(test_df)
        # Get first 15 characters from long TensorRT layer names
        test_df.loc[:, "layer_name"] = test_df.loc[:, "layer_name"].str[:15]
        ax = test_df.plot(rot=90, x="layer_name", kind="bar")
        return ax.get_figure()
