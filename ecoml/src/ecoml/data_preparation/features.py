"""Build input features from PyTorch model summary."""

# from data_preparation.pytorch_utils import PytorchLayer
from ecoml.data_preparation.pytorch_utils import PytorchLayer


def get_convolutional_features(layer_info: PytorchLayer) -> dict[str, int]:
    """Get features for a convolutional layer."""
    return {
        "batch_size": layer_info.input_shape[0],
        "input_size_0": layer_info.input_shape[1],  # We skip batch size
        "input_size_1": layer_info.input_shape[2],
        "input_size_2": layer_info.input_shape[3],
        "output_size_0": layer_info.output_shape[1],  # We skip batch size
        "output_size_1": layer_info.output_shape[2],
        "output_size_2": layer_info.output_shape[3],
        "kernel_0": layer_info.kernel_size[0],
        "kernel_1": layer_info.kernel_size[1],
        "padding_0": layer_info.padding[0],
        "padding_1": layer_info.padding[1],
        "stride_0": layer_info.stride[0],
        "stride_1": layer_info.stride[1],
    }


def get_pooling_features(layer_info: PytorchLayer) -> dict[str, int]:
    """Get features for a pooling layer."""
    if layer_info.kernel_size is None:  # Assume this is global pooling
        kernel_size = [
            layer_info.input_shape[2],
            layer_info.input_shape[3],
        ]
    else:
        kernel_size = layer_info.kernel_size

    if layer_info.stride is None:  # Assume this is global pooling
        stride = [1, 1]
    else:
        stride = layer_info.stride

    return {
        "batch_size": layer_info.input_shape[0],
        "input_size_0": layer_info.input_shape[1],  # We skip batch size
        "input_size_1": layer_info.input_shape[2],
        "input_size_2": layer_info.input_shape[3],
        "output_size_0": layer_info.output_shape[1],  # We skip batch size
        "output_size_1": layer_info.output_shape[2],
        "output_size_2": layer_info.output_shape[3],
        "kernel_0": kernel_size[0],
        "kernel_1": kernel_size[1],
        "stride_0": stride[0],
        "stride_1": stride[1],
    }


def get_dense_features(layer_info: PytorchLayer) -> dict[str, int]:
    """Get features for a dense layer.

    We also consider conv 1x1 as a dense layer.
    """
    return {
        "batch_size": layer_info.input_shape[0],
        "input_size": layer_info.input_shape[1],
        "output_size": layer_info.output_shape[1],
    }
