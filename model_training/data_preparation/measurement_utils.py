"""Parse measurement data."""

from pathlib import Path

import pandas as pd

SELECT_COLUMNS = ["layer_name", "average_power", "average_run_time"]

RENAME_COLUMNS = {
    "layer_power_excluding_idle_power_micro_watt": "average_power",
    "layer_run_time": "average_run_time",
}


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
    if verify_per_layer_measurements(df, min_per_layer_measurement):
        df = get_average_measurement_data(df)
        df.rename(columns=RENAME_COLUMNS, inplace=True)
        return df.loc[:, SELECT_COLUMNS]
    else:
        return None


def verify_per_layer_measurements(
    df: pd.DataFrame, min_per_layer_measurement: int
) -> bool:
    """Verify all layers have sufficient samples of power and runtime.

    Args:
        df: Input data containing power, latency runtime per layer.
            This data is recorded for N cycles for each layer.
        min_per_layer_measurement: Minimum number of samples
            of power and runtime values to be present for each layer


    Returns:
        True if all layers have sufficient samples of power and runtime values.
    """
    power_col_name = "layer_power_excluding_idle_power_micro_watt"
    runtime_col_name = "layer_run_time"
    per_layer_power_measurements = (
        df.groupby(["layer_name"])[power_col_name].count().values
    )
    per_layer_runtime_measurements = (
        df.groupby(["layer_name"])[runtime_col_name].count().values
    )
    # Check if power and runtime samples pass sufficient threshold
    if all(per_layer_power_measurements >= min_per_layer_measurement) and all(
        per_layer_runtime_measurements >= min_per_layer_measurement
    ):
        return True
    return False


def get_average_measurement_data(df: pd.DataFrame) -> pd.DataFrame:
    """Get average run time and average power for each layer.

    Args:
        df: Input data containing power, latency runtime per layer.
            This data is recorded for N cycles for each layer.

    Returns:
        Averaged power and latency runtime per unique layer.
    """
    # Columns for which average is calculated
    avg_columns = list(RENAME_COLUMNS.keys())
    avg_df = df.groupby(["layer_name"])[avg_columns].mean().reset_index()
    return avg_df
