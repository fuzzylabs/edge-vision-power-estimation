from pathlib import Path

import pandas as pd
from tqdm import tqdm
from data_preparation.tensorrt_utils import TensorRTLayer


def get_convolutional_features(layer_info: TensorRTLayer) -> dict[str, int]:
    """Get features for a convolutional layer."""
    return {
        "batch_size": layer_info.inputs[0].dimensions[0],
        "input_size_0": layer_info.inputs[0].dimensions[1],  # We skip batch size
        "input_size_1": layer_info.inputs[0].dimensions[2],
        "input_size_2": layer_info.inputs[0].dimensions[3],
        "output_size_0": layer_info.outputs[0].dimensions[1],  # We skip batch size
        "output_size_1": layer_info.outputs[0].dimensions[2],
        "output_size_2": layer_info.outputs[0].dimensions[3],
        "kernel_0": layer_info.kernel[0],
        "kernel_1": layer_info.kernel[1],
        # To keep the same kind of features as Neural Power
        "padding_0": layer_info.pre_padding[0] + layer_info.post_padding[0],
        "padding_1": layer_info.pre_padding[1] + layer_info.post_padding[1],
        "stride_0": layer_info.stride[0],
        "stride_1": layer_info.stride[1],
    }


def get_pooling_features(layer_info: TensorRTLayer) -> dict[str, int]:
    """Get features for a pooling layer."""
    if layer_info.kernel is None:  # Assume this is global pooling
        kernel = [
            layer_info.inputs[0].dimensions[2],
            layer_info.inputs[0].dimensions[3],
        ]
    else:
        kernel = layer_info

    return {
        "batch_size": layer_info.inputs[0].dimensions[0],
        "input_size_0": layer_info.inputs[0].dimensions[1],  # We skip batch size
        "input_size_1": layer_info.inputs[0].dimensions[2],
        "input_size_2": layer_info.inputs[0].dimensions[3],
        "output_size_0": layer_info.outputs[0].dimensions[1],  # We skip batch size
        "output_size_1": layer_info.outputs[0].dimensions[2],
        "output_size_2": layer_info.outputs[0].dimensions[3],
        "kernel_0": kernel[0],
        "kernel_1": kernel[1],
        "stride_0": layer_info.stride[0],
        "stride_1": layer_info.stride[1],
    }


def get_dense_features(layer_info: TensorRTLayer) -> dict[str, int]:
    """Get features for a dense layer."""
    return {
        "batch_size": layer_info.inputs[0].dimensions[0],
        "input_size": layer_info.inputs[0].dimensions[1],  # We skip batch size
        "output_size": layer_info.outputs[0].dimensions[1],  # We skip batch size
    }


def convert_measurements_to_training_data(
    save_path: Path, layers_info: dict[str, TensorRTLayer], measurements: pd.DataFrame
) -> None:
    """Convert measurements into training data and save to files.

    Args:
        save_path (Path): Directory path to save the training data to.
        layers_info (dict[str, TensorRTLayer]): Information about the layers.
        measurements (pd.DataFrame): Runtime and power measurements.
    """
    results = {
        "convolutional": [],
        "pooling": [],
        "dense": [],
    }

    layers_ignored = set()

    for _, row in tqdm(measurements.iterrows(), total=measurements.shape[0]):
        layer_name = row.layer_name
        if layer_name not in layers_info:
            print(f"Layer {layer_name} is not found!")
            continue

        layer_info = layers_info[layer_name]
        layer_type = layer_info.get_layer_type()
        if layer_type == "convolutional":
            features = get_convolutional_features(layer_info)
            features["power"] = row.average_power
            features["runtime"] = row.average_run_time
            features["layer_name"] = layer_name
            results["convolutional"].append(features)
        elif layer_type == "pooling":
            features = get_pooling_features(layer_info)
            features["power"] = row.average_power
            features["runtime"] = row.average_run_time
            features["layer_name"] = layer_name
            results["pooling"].append(features)
        elif layer_type == "dense":
            features = get_dense_features(layer_info)
            features["power"] = row.average_power
            features["runtime"] = row.average_run_time
            features["layer_name"] = layer_name
            results["dense"].append(features)
        else:
            layers_ignored.add(layer_type)

    print(f"Layer types ignored: {layers_ignored}")

    save_path.mkdir(parents=True, exist_ok=True)

    for layer_type in results:
        filepath = save_path / f"{layer_type}.csv"
        if len(results[layer_type]) > 0:
            df = pd.DataFrame(results[layer_type])
            df.to_csv(filepath, index=False)
        else:
            print(f"Skipping {layer_type} as it is empty")
