"""Test DataPreprocessor class."""
import json
from pathlib import Path

import pytest

from data_preparation.data_preprocess import DataPreprocessor
from data_preparation.io_utils import parse_model_dir


@pytest.fixture
def data_preprocessor(tmp_path: Path) -> DataPreprocessor:
    """DataPreprocessor instance fixture.

    See `example_data/idling_power.json` file for the example idling power.
    A temporary is used for the results stored, if any.
    """
    idle_power_log_path = Path(__file__).parent / "example_data" / "idling_power.json"
    return DataPreprocessor(idle_power_log_path, tmp_path)


def test_compute_layer_metrics_by_cycle(data_preprocessor: DataPreprocessor) -> None:
    """Test compute_layer_metrics_by_cycle."""


    model_dir = Path(__file__).parent / "example_data" / "simple"
    power_log_file, trt_layer_latency_file, trt_engine_info_file = (
        parse_model_dir(model_dir)
    )

    with open(model_dir / "expected_metrics_by_cycle.json", "r") as file:
        expected_metrics_by_cycle = json.load(file)

    metrics_by_cycle = data_preprocessor.compute_layer_metrics_by_cycle(power_log_file, trt_layer_latency_file, trt_engine_info_file)
    assert metrics_by_cycle == expected_metrics_by_cycle