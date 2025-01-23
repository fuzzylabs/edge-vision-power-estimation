"""Test Pytorch utility functions."""

from data_preparation.pytorch_utils import (
    read_layers_info,
    PytorchLayer
)

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent / "sample_data"

def test_read_layers_info():
    """Test that `read_layers_info` returns a dictionary of `PytorchLayer`."""
    resnet18_sample_data_path = BASE_DIR / "resnet18_model_summary.json"
    layer_info = read_layers_info(resnet18_sample_data_path)

    assert isinstance(layer_info, dict)
    assert isinstance(layer_info["_conv1"], PytorchLayer)