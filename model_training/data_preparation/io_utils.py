"""Utility functions for IO."""

import json
from pathlib import Path


def read_json_file(file_path: str | Path) -> dict:
    """Read json file.

    Args:
        file_path: Path to json file

    Returns:
        Return the content as dict from json file.
    """
    with open(file_path, "r") as fp:
        return json.load(fp)


def read_log_file(file_path: str | Path) -> list[str]:
    """Read log file and return the contents from the file.

    Args:
        file_path: Path to file

    Returns:
        Contents as list of string from the file.
    """
    with open(file_path, "r") as fp:
        return fp.readlines()


def parse_model_dir(model_dir: str | Path) -> tuple[Path, Path, Path]:
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
    log_files = list(model_dir_path.glob("*.log"))

    if len(log_files) != 1:
        raise ValueError(
            f"Expected exactly 1 *.log file inside {model_dir_path}, found {len(log_files)}"
        )

    return (
        log_files[0],  # Power log file
        model_dir_path / "trt_profiling/trt_layer_latency.json",
        model_dir_path / "trt_profiling/trt_engine_info.json",
    )


def get_idle_power_log_file(raw_data_dir: str) -> Path:
    """Get path to the power log file inside raw data directory.

    It is of format "{n}_seconds_idling_power_log_{timestamp}.log".

    Args:
        raw_data_dir: Path to raw dataset directory.

    Raises:
        ValueError: If there are zero or more than 1 log files inside
            raw data directory.

    Returns:
        Path to the power log file for idle state.
    """
    raw_data_path = Path(raw_data_dir)
    log_files = list(raw_data_path.glob("*.log"))

    if len(log_files) != 1:
        raise ValueError(
            f"Expected exactly 1 *.log file inside {raw_data_path}, found {len(log_files)}"
        )

    return log_files[0]
