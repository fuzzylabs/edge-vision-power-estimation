"""Utility functions for IO."""

import json
from pathlib import Path
from typing import Any

import yaml


def read_yaml_file(file_path: Path) -> Any:
    """Read yaml file.

    Args:
        file_path (Path): The filepath to load from.

    Returns:
        Any: Configuration
    """
    with open(file_path) as fp:
        config = yaml.safe_load(fp)
    return config


def read_json_file(file_path: Path) -> dict:
    """Read json file.

    Args:
        file_path: Path to json file

    Returns:
        Return the content as dict from json file.
    """
    with open(file_path, "r") as fp:
        return json.load(fp)


def read_log_file(file_path: Path) -> list[str]:
    """Read log file and return the contents from the file.

    Args:
        file_path: Path to file

    Returns:
        Contents as list of string from the file.
    """
    with open(file_path, "r") as fp:
        return fp.readlines()


def parse_model_dir(model_dir: Path) -> tuple[Path, Path, Path]:
    """Get path to relevant files from model directory.

    Args:
        model_dir: Path to model directory.

    Raises:
        ValueError: If there are zero or more than 1 files ending with
            *.log extension in model directory.

    Returns:
        Tuple of path to log file containing timestamped power values,
        timestamped runtime values inside trt_layer_latency.json file,
        and tensorrt engine info inside trt_engine_info.json file.
    """
    model_dir_path = Path(model_dir)
    model_name = model_dir_path.stem
    return (
        model_dir_path / f"{model_name}_power_log.log",
        model_dir_path / "trt_profiling/trt_layer_latency.json",
        model_dir_path / "trt_profiling/trt_engine_info.json",
    )


def get_idle_power_log_file(raw_data_dir: Path) -> Path:
    """Get path to the power log file inside raw data directory.

    The name of json file inside raw data directory
    is `idling_power.json`.

    Args:
        raw_data_dir: Path to raw dataset directory.

    Returns:
        Path to the power log file for idle state.
    """
    return f"{raw_data_dir}/idling_power.json"
