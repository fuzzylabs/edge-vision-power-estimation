"""Test measurement_utils module."""

from pathlib import Path

import pandas as pd
import pytest

from data_preparation.measurement_utils import (
    LAYER_TYPES,
    RENAME_COLUMNS,
    preprocess_measurement_data,
    verify_per_layer_measurements,
)

BASE_DIR = Path(__file__).parent.parent / "test_data/example_preprocessed_data"

EXPECTED_POWER_MODEL1 = [4.5, 5.0]
EXPECTED_RUNTIME_MODEL1 = [0.08, 0.01]
EXPECTED_POWER_MODEL2 = [3.75, 4.75]
EXPECTED_RUNTIME_MODEL2 = [0.085, 0.01]


@pytest.mark.parametrize(
    ["input_path", "min_samples", "expected_output"],
    [
        ("model1", 1, True),
        ("model1", 2, False),
        ("model1", 3, False),
        ("model2", 1, True),
        ("model2", 2, True),
        ("model2", 3, True),
    ],
)
def test_verify_per_layer_power_measurements(input_path, min_samples, expected_output):
    """Test verify_per_layer_measurements function.

    It return True if there are enough samples for power and runtime measurements.

    Args:
        input_path: Path to preprocessed data csv
        min_samples: Min samples for power and runtime
        expected_output: Expected output
    """
    path = BASE_DIR / input_path / "power_runtime_mapping_layerwise.csv"
    df = pd.read_csv(path)
    df.rename(columns=RENAME_COLUMNS, inplace=True)
    df = df[df["layer_type"].isin(LAYER_TYPES)]
    out = verify_per_layer_measurements(df, min_samples, "average_power")
    assert expected_output == out


@pytest.mark.parametrize(
    ["input_path", "min_samples", "expected_output"],
    [
        ("model1", 1, True),
        ("model1", 2, True),
        ("model1", 3, True),
        ("model2", 1, True),
        ("model2", 2, True),
        ("model2", 3, True),
    ],
)
def test_verify_per_layer_runtime_measurements(
    input_path, min_samples, expected_output
):
    """Test verify_per_layer_measurements function.

    It return True if there are enough samples for power measurements.

    Args:
        input_path: Path to preprocessed data csv
        min_samples: Min samples for power and runtime
        expected_output: Expected output
    """
    path = BASE_DIR / input_path / "power_runtime_mapping_layerwise.csv"
    df = pd.read_csv(path)
    df.rename(columns=RENAME_COLUMNS, inplace=True)
    df = df[df["layer_type"].isin(LAYER_TYPES)]
    out = verify_per_layer_measurements(df, min_samples, "average_run_time")
    assert expected_output == out


@pytest.mark.parametrize(
    ["input_path", "min_samples", "expected_output"],
    [
        ("model1", 1, (EXPECTED_POWER_MODEL1, EXPECTED_RUNTIME_MODEL1)),
        ("model2", 1, (EXPECTED_POWER_MODEL2, EXPECTED_RUNTIME_MODEL2)),
        ("model2", 2, (EXPECTED_POWER_MODEL2, EXPECTED_RUNTIME_MODEL2)),
        ("model2", 3, (EXPECTED_POWER_MODEL2, EXPECTED_RUNTIME_MODEL2)),
    ],
)
def test_preprocess_measurement_data(input_path, min_samples, expected_output):
    """Test preprocess_measurement_data return expected power and runtime values.

    Args:
        input_path: Path to preprocessed data csv
        min_samples: Min samples for power and runtime
        expected_output: Expected output containing tuple of power and runtime output
    """
    path = BASE_DIR / input_path / "power_runtime_mapping_layerwise.csv"
    path = Path(path)
    out = preprocess_measurement_data(path, min_samples)
    assert expected_output[0] == out.average_power.values.tolist()
    assert expected_output[1] == out.average_run_time.values.tolist()


@pytest.mark.parametrize(
    ["input_path", "min_samples", "expected_output"],
    [
        ("model1", 2, None),
        ("model1", 3, None),
    ],
)
def test_preprocess_measurement_data_empty_return(
    input_path, min_samples, expected_output
):
    """Test preprocess_measurement_data return None if not enough samples.

    Args:
        input_path: Path to preprocessed data csv
        min_samples: Min samples for power and runtime
        expected_output: Expected output
    """
    path = BASE_DIR / input_path / "power_runtime_mapping_layerwise.csv"
    path = Path(path)
    out = preprocess_measurement_data(path, min_samples)
    assert expected_output == out
