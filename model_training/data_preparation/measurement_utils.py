"""Parse measurement data."""

from pathlib import Path

import pandas as pd
from loguru import logger

SELECT_COLUMNS = ["layer_name", "average_power", "average_run_time"]

RENAME_COLUMNS = {
    "layer_power_excluding_idle_power_micro_watt": "average_power",
    "layer_run_time": "average_run_time",
}

LAYER_TYPES = ["CaskPooling", "CaskConvolution", "CaskGemmConvolution"]


def preprocess_measurement_data(
    path: Path, min_per_layer_measurement: int
) -> pd.DataFrame | None:
    """Read and preprocess measurement data from CSV file.

    For each layer, we have N records of power and latency.
    We get average for power and latency for each layer over N cycles.

    Args:
        path: Path to csv file
        min_per_layer_measurement: Minimum number of samples
            of power and runtime values to be present for each layer

    Returns:
        Parse csv data as pandas dataframe.
    """
    df = pd.read_csv(path)
    df.rename(columns=RENAME_COLUMNS, inplace=True)
    filtered_df = df[df["layer_type"].isin(LAYER_TYPES)]

    # Get mean of power and runtime
    # Also save the statistics such as count, mean to a CSV
    avg_df = get_average_measurement_data(path, filtered_df)

    # Verify if power and runtime samples exceed the threshold
    valid_power_samples = verify_per_layer_measurements(
        filtered_df, min_per_layer_measurement, "average_power"
    )
    valid_runtime_samples = verify_per_layer_measurements(
        filtered_df, min_per_layer_measurement, "average_run_time"
    )
    if valid_power_samples and valid_runtime_samples:
        return avg_df.loc[:, SELECT_COLUMNS]
    else:
        return None


def verify_per_layer_measurements(
    df: pd.DataFrame, min_per_layer_measurement: int, col_name: str
) -> bool:
    """Verify all layers have sufficient samples of power and runtime.

    Args:
        df: Input data containing power, latency runtime per layer.
            This data is recorded for N cycles for each layer.
        min_per_layer_measurement: Minimum number of samples
            of power and runtime values to be present for each layer
        col_name: Name of column to get sample count for each layer

    Returns:
        True if all layers have sufficient samples of power and runtime values.
    """
    count_df = df.groupby(["layer_name"])[col_name].count().reset_index()
    less_samples_df = count_df[count_df[col_name] < min_per_layer_measurement]
    if len(less_samples_df):
        logger.debug(
            f"Following {len(less_samples_df)} layers do not meet the "
            f"threshold of {min_per_layer_measurement} samples"
        )
        for _, row in less_samples_df.iterrows():
            logger.debug(f"{row['layer_name']} -> {row[col_name]}")
    # Check if power or runtime samples pass sufficient threshold
    if all(count_df[col_name] >= min_per_layer_measurement):
        return True
    return False


def get_average_measurement_data(path: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Get average run time and average power for each layer.

    This function also calculates statistics such count, mean
    for power and runtime and saves to a CSV file.

    Args:
        path: Path to csv file
        df: Input data containing power, latency runtime per layer.
            This data is recorded for N cycles for each layer.

    Returns:
        Averaged power and latency runtime per unique layer.
    """
    # Columns for which average is calculated
    avg_columns = list(RENAME_COLUMNS.values())
    avg_df = df.groupby(["layer_name"])[avg_columns].mean().reset_index()
    model_dir = path.parent
    model_name = model_dir.name
    for col in avg_columns:
        save_stats_csv_path = f"{model_dir}/{model_name}_{col}_stats.csv"
        df.groupby(["layer_name"])[col].describe().to_csv(save_stats_csv_path)
        logger.info(f"Saved layer stastics to {save_stats_csv_path} csv file")
    return avg_df
