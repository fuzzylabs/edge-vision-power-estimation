"""Test convert function used in data preprocessing."""

import json
from pathlib import Path

import pytest

from data_preparation.convert import (
    get_convolutional_features,
    get_dense_features,
    get_pooling_features,
)
from data_preparation.tensorrt_utils import read_layers_info

BASE_DIR = Path(__file__).parent / "example_preprocessed_data"


@pytest.fixture
def layers_info(request):
    """Parse TRT engine info file to get layer information."""
    path = Path(f"{BASE_DIR}/{request.param}/trt_engine_info.json")
    layer_info = read_layers_info(path)
    return layer_info


@pytest.fixture
def expected_features(request):
    """Expected features."""
    expected_json_file = Path(f"{BASE_DIR}/{request.param}/expected_features.json")
    with open(expected_json_file, "r") as fp:
        return json.load(fp)


@pytest.mark.parametrize(
    ("layer", "layers_info", "expected_features"),
    [("conv_relu_layer", "model1", "model1"), ("conv_relu_layer", "model2", "model2")],
    indirect=("layers_info", "expected_features"),
)
def test_get_convolutional_features(layer, layers_info, expected_features):
    """Test convolutional features are parsed correctly."""
    layer_info = layers_info[layer]
    layer_type = layer_info.get_layer_type()

    assert layer_type == "convolutional"

    if layer_type == "convolutional":
        features = get_convolutional_features(layer_info)
        assert expected_features["convolutional"] == features


@pytest.mark.parametrize(
    ("layer", "layers_info", "expected_features"),
    [("maxpool_layer", "model1", "model1"), ("maxpool_layer", "model2", "model2")],
    indirect=("layers_info", "expected_features"),
)
def test_get_pooling_features(layer, layers_info, expected_features):
    """Test pooling features are parsed correctly."""
    layer_info = layers_info[layer]
    layer_type = layer_info.get_layer_type()

    assert layer_type == "pooling"

    if layer_type == "pooling":
        features = get_pooling_features(layer_info)
        assert expected_features["pooling"] == features


@pytest.mark.parametrize(
    ("layer", "layers_info", "expected_features"),
    [("dense_layer", "model1", "model1"), ("dense_layer", "model2", "model2")],
    indirect=("layers_info", "expected_features"),
)
def test_get_dense_layers(layer, layers_info, expected_features):
    """Test pooling features are parsed correctly."""
    layer_info = layers_info[layer]
    layer_type = layer_info.get_layer_type()

    assert layer_type == "dense"

    if layer_type == "dense":
        features = get_dense_features(layer_info)
        assert expected_features["dense"] == features
