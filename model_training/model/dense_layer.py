from pathlib import Path

from sklearn.pipeline import Pipeline
import pandas as pd

from model.common import create_pipeline, turn_into_mapping, read_data

features = [
    "batch_size",
    "input_size",
    "output_size",
]

features_mapping = turn_into_mapping(features)


def read_dense_data(path: Path) -> (pd.DataFrame, pd.Series, pd.Series):
    """Read data for a dense layer."""
    return read_data(path, features)


def create_power_pipeline() -> Pipeline:
    """Create dense layer power prediction pipeline."""
    return create_pipeline(
        features_mapping,
        polynomial_degree=3,
    )


def create_runtime_pipeline() -> Pipeline:
    """Create dense layer runtime prediction pipeline."""
    return create_pipeline(
        features_mapping,
        polynomial_degree=3,
    )
