from pathlib import Path

from sklearn.pipeline import Pipeline
import pandas as pd

from model_training.model.common import create_pipeline, turn_into_mapping, read_data

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
    "padding_0",
    "padding_1",
    "stride_0",
    "stride_1"
]

features_mapping = turn_into_mapping(features)

TOTAL_OPS_PER_INPUT = [
    "input_size_0", "input_size_1", "input_size_2",
    "kernel_0", "kernel_1", "output_size_2",
]
TOTAL_OPS_PER_BATCH = TOTAL_OPS_PER_INPUT + ["batch_size"]


def read_convolutional_data(path: Path) -> (pd.DataFrame, pd.Series, pd.Series):
    """Read data for a convolutional layer."""
    return read_data(path, features)


def create_power_pipeline() -> Pipeline:
    """Create convolutional layer power prediction pipeline."""
    return create_pipeline(
        features_mapping,
        polynomial_degree=2,
        is_log=True,
        special_terms_list=[TOTAL_OPS_PER_INPUT, TOTAL_OPS_PER_BATCH]
    )


def create_runtime_pipeline() -> Pipeline:
    """Create convolutional layer runtime prediction pipeline."""
    return create_pipeline(
        features_mapping,
        polynomial_degree=3,
        special_terms_list=[TOTAL_OPS_PER_INPUT, TOTAL_OPS_PER_BATCH]
    )
