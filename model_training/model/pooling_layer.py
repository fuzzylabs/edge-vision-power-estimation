from pathlib import Path

import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline

from model.common import (
    create_dataset,
    create_pipeline,
    eval_metrics,
    plot_layerwise_predictions,
    train_test_split,
    turn_into_mapping,
)

POOLING_FEATURES = [
    "batch_size",
    "input_size_0",
    "input_size_1",
    "input_size_2",
    "output_size_0",
    "output_size_1",
    "output_size_2",
    "kernel_0",
    "kernel_1",
    "stride_0",
    "stride_1",
]

POOLING_FEATURES_MAPPING = turn_into_mapping(POOLING_FEATURES)

TOTAL_INPUT_FEATURES = ["batch_size", "input_size_0", "input_size_1", "input_size_2"]

TOTAL_OUTPUT_FEATURES = [
    "batch_size",
    "output_size_0",
    "output_size_1",
    "output_size_2",
]

TOTAL_NO_OPS = [
    "batch_size",
    "input_size_0",
    "input_size_1",
    "input_size_2",
    "kernel_0",
    "kernel_1",
]


def create_pooling_dataset(
    csv_paths: list[Path], features: list[str] = POOLING_FEATURES
) -> tuple[pd.DataFrame, pd.Series, pd.Series] | None:
    """Create data for a pooling layer CSVs.

    Args:
        csv_paths: Path to pooling layer data CSVs.
        features: List of pooling layer feature columns.
            Defaults to POOLING_FEATURES.

    Returns:
        Tuple of Dataframe containing input features,
        pandas series containing power values
        pandas series containing runtime values
    """
    return create_dataset(csv_paths, features)


def create_power_pipeline(
    features_mapping=POOLING_FEATURES_MAPPING, degree: int = 3
) -> Pipeline:
    """Create pooling layer power prediction pipeline.

    Args:
        features_mapping: Dictionary mapping feature name to it's index.
            Defaults to POOLING_FEATURES_MAPPING.
        degree: Polynomial degree. Defaults to 3.

    Returns:
        sklearn pipeline
    """
    return create_pipeline(
        features_mapping,
        polynomial_degree=degree,
        special_terms_list=[TOTAL_NO_OPS, TOTAL_INPUT_FEATURES, TOTAL_OUTPUT_FEATURES],
    )


def create_runtime_pipeline(
    features_mapping=POOLING_FEATURES_MAPPING, degree: int = 3
) -> Pipeline:
    """Create pooling layer runtime prediction pipeline.

    Args:
        features_mapping: Dictionary mapping feature name to it's index.
            Defaults to POOLING_FEATURES_MAPPING.
        degree: Polynomial degree. Defaults to 3.

    Returns:
        sklearn pipeline
    """
    return create_pipeline(
        features_mapping,
        polynomial_degree=degree,
        special_terms_list=[TOTAL_NO_OPS, TOTAL_INPUT_FEATURES, TOTAL_OUTPUT_FEATURES],
    )


def pooling_pipeline(data_config: dict, features: list[str] = POOLING_FEATURES):
    """Train power and runtime model using pooling layer data.

    Args:
        data_config: Configuration for the dataset
        features: List of pooling layer feature columns.
            Defaults to POOLING_FEATURES.
    """
    print("Training model using pooling layer data")
    data_dir = Path(data_config["data_dir"])
    test_models = data_config["test_models"]

    train_paths, test_paths = train_test_split(
        data_dir=data_dir, test_models=test_models, pattern="**/pooling.csv"
    )
    if train_paths is None or test_paths is None:
        print("No data found for training")
        return

    input_features_train, power_train, runtime_train = create_pooling_dataset(
        train_paths
    )
    input_features_test, power_test, runtime_test = create_pooling_dataset(test_paths)

    print(f"Training samples: {len(input_features_train)}")
    print(f"Testing samples: {len(input_features_test)}")

    print("Training power model")
    with mlflow.start_run(run_name="pooling_power_model"):
        power_pipeline = create_power_pipeline()
        power_pipeline.fit(input_features_train.values, power_train.values)
        power_pred = power_pipeline.predict(input_features_test.values)
        test_metrics = eval_metrics(actual=power_test, pred=power_pred)
        mlflow.log_metrics(test_metrics)
        for test_file_path in test_paths:
            model_name = test_file_path.parent.stem
            fig = plot_layerwise_predictions(
                test_file_path=test_file_path,
                features=features,
                model=power_pipeline,
                model_type="power",
            )
            fig.tight_layout()
            mlflow.log_figure(fig, f"{model_name}_pooling_power_prediction.png")

    print("Training runtime model")
    with mlflow.start_run(run_name="pooling_runtime_model"):
        runtime_pipeline = create_runtime_pipeline()
        runtime_pipeline.fit(input_features_train.values, runtime_train.values)
        runtime_pred = runtime_pipeline.predict(input_features_test.values)
        test_metrics = eval_metrics(actual=runtime_test, pred=runtime_pred)
        mlflow.log_metrics(test_metrics)
        for test_file_path in test_paths:
            model_name = test_file_path.parent.stem
            fig = plot_layerwise_predictions(
                test_file_path=test_file_path,
                features=features,
                model=runtime_pipeline,
                model_type="runtime",
            )
            fig.tight_layout()
            mlflow.log_figure(fig, f"{model_name}_pooling_runtime_prediction.png")
