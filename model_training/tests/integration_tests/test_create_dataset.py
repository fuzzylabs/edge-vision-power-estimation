"""Integration test for preprocessed and training data."""

import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from convert_measurements import main as get_training_data
from map_power_to_layers import main as get_preprocessed_data

BASE_DIR = Path(__file__).parent.parent / "test_data"


@pytest.fixture(scope="session")
def shared_temp_dir(tmp_path_factory):
    """Path to shared temp directory."""
    temp_dir = tmp_path_factory.mktemp("temp_data")
    return temp_dir


@pytest.fixture
def preprocessed_dataset_args(shared_temp_dir):
    """Arguments passed for creating preprocessed dataset."""
    preprocessed_data_path = os.path.join(shared_temp_dir, "preprocessed_data")
    raw_data_path = f"{BASE_DIR}/example_raw_data"
    return SimpleNamespace(
        result_dir=preprocessed_data_path, raw_data_dir=raw_data_path
    )


@pytest.fixture
def training_dataset_args(shared_temp_dir):
    """Arguments passed for creating training dataset."""
    training_data_path = os.path.join(shared_temp_dir, "training_data")
    preprocessed_data_path = os.path.join(shared_temp_dir, "preprocessed_data")
    return SimpleNamespace(
        result_dir=training_data_path,
        preprocessed_data_dir=preprocessed_data_path,
        per_layer_measurements=1,
    )


def verify_preprocessed_dataset(shared_temp_dir):
    """Verify the files and data for preprocessed data is created.

    Args:
        shared_temp_dir: Path to temp directory.
    """
    preprocessed_data_path = os.path.join(shared_temp_dir, "preprocessed_data")
    expected_preprocessed_data_folder = sorted(
        ["multiple_layers", "simple", "multiple_readings"]
    )
    assert os.path.exists(preprocessed_data_path)
    assert (
        sorted(os.listdir(preprocessed_data_path)) == expected_preprocessed_data_folder
    )

    for folder in os.listdir(preprocessed_data_path):
        model_dir = os.path.join(preprocessed_data_path, folder)
        files = os.listdir(model_dir)
        assert len(files) == 2

        assert sorted(files) == sorted(
            ["power_runtime_mapping_layerwise.csv", "trt_engine_info.json"]
        )

        output_preprocessed_csv = os.path.join(
            model_dir, "power_runtime_mapping_layerwise.csv"
        )
        expected_output = os.path.join(
            BASE_DIR, "example_raw_data", folder, "expected_metrics_by_cycle.json"
        )

        df_output = pd.read_csv(output_preprocessed_csv)
        # JSON converts layer_runtime dtype to int64?
        df_expected = pd.read_json(expected_output)
        pd.testing.assert_frame_equal(df_output, df_expected, check_dtype=False)


def verify_training_dataset(shared_temp_dir):
    """Verify the files and data for training data is created.

    Args:
        shared_temp_dir: Path to temp directory.
    """
    training_data_path = os.path.join(shared_temp_dir, "training_data")
    expected_training_data_folder = sorted(
        ["multiple_layers", "simple", "multiple_readings"]
    )
    assert os.path.exists(training_data_path)
    assert sorted(os.listdir(training_data_path)) == expected_training_data_folder

    for folder in os.listdir(training_data_path):
        model_dir = os.path.join(training_data_path, folder)
        files = os.listdir(model_dir)
        # Can include any of the layer type
        assert len(files) <= 3

        # Not all models have all the layer types
        expected_training_files = set(["convolutional.csv", "pooling.csv", "dense.csv"])
        assert set(files).issubset(expected_training_files)


def test_create_dataset(
    shared_temp_dir, preprocessed_dataset_args, training_dataset_args
):
    """Integration test.

    We create a test that tests our workflow for generating
    training dataset from raw dataset. There are two steps

    1. Using raw dataset, generate preprocessed dataset
    using `map_power_to_layers.py`
    2. Using preprocessed data, create training dataset
    using `convert_measurements.py`
    """
    get_preprocessed_data(preprocessed_dataset_args)

    verify_preprocessed_dataset(shared_temp_dir)

    get_training_data(training_dataset_args)

    verify_training_dataset(shared_temp_dir)
