from pathlib import Path
import pandas as pd

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
import numpy as np


MICROWATTS_IN_WATTS = 1e6

CROSS_VALIDATION = 10

def turn_into_mapping(column_names: list[str]) -> dict[str, int]:
    """Turn a list of features into a mapping from feature names to index."""
    return {feature: i for i, feature in enumerate(column_names)}


def read_data(
    file_path: Path, features: list[str]
) -> (pd.DataFrame, pd.Series, pd.Series):
    """Read data for a layer.

    Args:
        file_path (Path): Path to the CSV data file.
        features (list[str]): List of feature column names.

    Returns:
        pd.DataFrame: input features
        pd.Series: measured power
        pd.Series: measured runtime
    """
    df = pd.read_csv(file_path)

    input_features = df.loc[:, features]
    # Convert microwatt to watt
    power = df.power / MICROWATTS_IN_WATTS
    # Recorded in milliseconds
    runtime = df.runtime

    return input_features, power, runtime


def _create_pipeline(transformer: TransformerMixin, cv: int = CROSS_VALIDATION) -> Pipeline:
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
