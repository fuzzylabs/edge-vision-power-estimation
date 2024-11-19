"""Common function shared."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.linear_model import LassoCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures

MICROWATTS_IN_WATTS = 1e6

CROSS_VALIDATION = 10


def turn_into_mapping(column_names: list[str]) -> dict[str, int]:
    """Turn a list of features into a mapping from feature names to index."""
    return {feature: i for i, feature in enumerate(column_names)}


def read_data(file_path: Path) -> pd.DataFrame:
    """Read data for a CSV file.

    It converts power from microwatt to watt.

    Args:
        file_path: Path to CSV file.

    Returns:
        Dataframe
    """
    df = pd.read_csv(file_path)
    # Convert microwatt to watt
    df.power = df.power / MICROWATTS_IN_WATTS
    return df


def create_dataset(
    file_paths: list[Path], features: list[str]
) -> tuple[pd.DataFrame, pd.Series, pd.Series] | None:
    """Create a dataset using list of path to CSV.

    Args:
        file_path: A list of paths to the CSV data file.
        features: List of feature column names.

    Returns:
        pd.DataFrame: input features
        pd.Series: measured power
        pd.Series: measured runtime
    """
    data = []
    for file_path in file_paths:
        df = read_data(file_path=file_path)
        data.append(df)

    df = pd.concat(data)
    input_features = df.loc[:, features]
    # Converted to watt from microwatt
    power = df.power
    # Recorded in milliseconds
    runtime = df.runtime
    return input_features, power, runtime


def _create_pipeline(
    transformer: TransformerMixin, cv: int = CROSS_VALIDATION
) -> Pipeline:
    """Create a neural power pipeline with given transformer."""
    return Pipeline([("transformer", transformer), ("lasso", LassoCV(cv=cv))])


def create_pipeline(
    features_mapping: dict[str, int],
    polynomial_degree: int,
    is_log: bool = False,
    special_terms_list: list[list[str]] | None = None,
) -> Pipeline:
    """Create a neural power pipeline.

    Args:
        features_mapping (dict[str, int]): Mapping of feature names to indices.
        polynomial_degree (int): Polynomial degree of regular polynomial terms.
        is_log (bool): Whether to log1p input features.
        special_terms_list (list[list[str]]): Definitions of special polynomial terms.

    Returns:
        Pipeline: neural power pipeline
    """
    regular_terms_transformer = get_regular_polynomial_terms_transformer(
        degree=polynomial_degree, is_log=is_log
    )
    if special_terms_list is not None and len(special_terms_list) > 0:
        special_terms_transformer = get_special_polynomial_terms_transformer(
            features_mapping,
            special_terms_list,
        )
        transformer = FeatureUnion(
            [
                ("regular_polynomial", regular_terms_transformer),
                ("special_polynomial", special_terms_transformer),
            ]
        )
    else:
        transformer = regular_terms_transformer

    return _create_pipeline(transformer)


def multiply_columns(input_arr: np.ndarray, column_indices: list[int]) -> np.ndarray:
    """Multiple all specified columns row-wise."""
    return np.multiply.reduce(input_arr[:, column_indices], axis=1)


def get_regular_polynomial_terms_transformer(
    degree: int, is_log: bool = False
) -> Pipeline | TransformerMixin:
    """Create a polynomial terms transformer."""
    polynomial_transformer = PolynomialFeatures(degree=degree)
    if is_log:
        return Pipeline(
            [
                # log2 produces -inf if any input feature (e.g. padding_0 or padding_1) is zero
                ("log1p", FunctionTransformer(np.log1p)),
                ("polynomial_features", polynomial_transformer),
            ]
        )
    else:
        return polynomial_transformer


def get_special_polynomial_terms_transformer(
    features_mapping: dict[str, int], terms_list: list[list[str]]
) -> TransformerMixin:
    """Create a transformer for special polynomial terms."""

    def _build_special_terms(input_arr: np.ndarray) -> np.ndarray:
        """Build special terms based on the mappings and the terms list."""
        result = []
        for terms in terms_list:
            column_indices = [features_mapping[t] for t in terms]
            result.append(multiply_columns(input_arr, column_indices))

        return np.stack(result).T

    return FunctionTransformer(_build_special_terms)


def train_test_split(
    data_dir: Path, test_models: list[str], pattern: str
) -> tuple[list[Path] | None, list[Path] | None]:
    """Split dataset into train and test sets.

    Args:
        data_dir: Path to training dataset
        test_models: List of models to use as test set
        pattern: Pattern to find relevant CSV data files.

    Returns:
        Tuple of train and test sets.
    """
    csv_paths = list(data_dir.rglob(pattern))
    # Return None if there are no files matching the pattern
    if not len(csv_paths):
        return None, None

    train_paths, test_paths = [], []
    for file in csv_paths:
        # Get name of model from path
        model_name = file.parent.stem
        if model_name not in test_models:
            train_paths.append(file)
        else:
            test_paths.append(file)
    return train_paths, test_paths


def eval_metrics(actual, pred, prefix: str = "testing_") -> dict[str, float]:
    """Calculate evaluation metrics.

    Args:
        actual: Actual values
        pred: Predicted values

    Returns:
        Dictionary mapping metric name to it's score.
    """
    EPSILON = 1e-10
    rmspe = np.sqrt(np.mean(np.square((actual - pred) / (actual + EPSILON)))) * 100
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
    test_file_path: Path, features: list[str], model, model_type: str
) -> plt.figure:
    """Plot layerwise prediction for given model and test dataset.

    Args:
        test_file_path: Path to test CSV file.
        features: List of feature column names.
        model: Trained sklearn model
        model_type: Type of trained model.

    Returns:
        Matplotlib figure.
    """
    test_df = read_data(file_path=test_file_path)
    pred = model.predict(test_df[features].values)
    test_df[f"{model_type}_pred"] = pred
    test_df = test_df[["layer_name", f"{model_type}", f"{model_type}_pred"]]
    print(f"Predictions for {test_file_path.parent.stem} model using {model_type}")
    print(test_df)
    # Get first 15 characters from long TensorRT layer names
    test_df.loc[:, "layer_name"] = test_df.loc[:, "layer_name"].str[:15]
    ax = test_df.plot(rot=90, x="layer_name", kind="bar")
    return ax.get_figure()
