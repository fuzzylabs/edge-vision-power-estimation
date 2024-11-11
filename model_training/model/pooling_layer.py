from pathlib import Path

from sklearn.pipeline import Pipeline
import pandas as pd

from model.common import create_pipeline, turn_into_mapping, read_data

features = [
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

features_mapping = turn_into_mapping(features)

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


def read_pooling_data(path: Path) -> (pd.DataFrame, pd.Series, pd.Series):
    """Read data for a pooling layer."""
    return read_data(path, features)


def create_power_pipeline() -> Pipeline:
    """Create pooling layer power prediction pipeline."""
    return create_pipeline(
        features_mapping,
        polynomial_degree=3,
        special_terms_list=[TOTAL_NO_OPS, TOTAL_INPUT_FEATURES, TOTAL_OUTPUT_FEATURES],
    )


def create_runtime_pipeline() -> Pipeline:
    """Create pooling layer runtime prediction pipeline."""
    return create_pipeline(
        features_mapping,
        polynomial_degree=3,
        special_terms_list=[TOTAL_NO_OPS, TOTAL_INPUT_FEATURES, TOTAL_OUTPUT_FEATURES],
    )
