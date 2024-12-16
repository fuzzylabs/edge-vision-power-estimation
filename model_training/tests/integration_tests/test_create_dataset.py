"""Integration test for preprocessed and training data."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from convert_measurements import main as get_training_data
from map_power_to_layers import main as get_preprocessed_data

BASE_DIR = Path(__file__).parent.parent / "test_data"


@pytest.fixture(scope="session")
def shared_temp_dir(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("temp_data")
    return temp_dir


@pytest.fixture
def preprocessed_dataset_args(shared_temp_dir):
    preprocessed_data_path = os.path.join(shared_temp_dir, "preprocessed_data")
    raw_data_path = f"{BASE_DIR}/example_raw_data"
    return SimpleNamespace(
        result_dir=preprocessed_data_path, raw_data_dir=raw_data_path
    )


@pytest.fixture
def training_dataset_args(shared_temp_dir):
    training_data_path = os.path.join(shared_temp_dir, "training_data")
    preprocessed_data_path = os.path.join(shared_temp_dir, "preprocessed_data")
    return SimpleNamespace(
        result_dir=training_data_path,
        preprocessed_data_dir=preprocessed_data_path,
        per_layer_measurements=1,
    )


def verify_preprocessed_dataset(shared_temp_dir):
    preprocessed_data_path = os.path.join(shared_temp_dir, "preprocessed_data")
    expected_preprocessed_data_folder = sorted(
        ["multiple_layers", "simple", "multiple_readings"]
    )
    assert os.path.exists(preprocessed_data_path)
    assert (
        sorted(os.listdir(preprocessed_data_path)) == expected_preprocessed_data_folder
    )


def verify_training_dataset(shared_temp_dir):
    training_data_path = os.path.join(shared_temp_dir, "training_data")
    expected_training_data_folder = sorted(
        ["multiple_layers", "simple", "multiple_readings"]
    )
    assert os.path.exists(training_data_path)
    assert sorted(os.listdir(training_data_path)) == expected_training_data_folder


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
