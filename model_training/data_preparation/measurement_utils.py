"""Parse measurement data."""

from pathlib import Path
import pandas as pd


SELECT_COLUMNS = [
    "layer_name",
    "average_power",
    "average_run_time",
]
RENAME_COLUMNS = {
    "layer_power_excluding_idle_power_micro_watt": "average_power",
    "layer_run_time": "average_run_time",
}


def preprocess_measurement_data(path: Path) -> pd.DataFrame:
    """Read and preprocess measurement data from CSV file.

    For each layer, we have N records of power and latency.
    We get average for power and latency for each layer over N cycles.

    Args:
        path: Path to csv file

    Returns:
        Parse csv data as pandas dataframe.
    """
    df = pd.read_csv(path)
    df = process_measurement_data(df)
    df.rename(columns=RENAME_COLUMNS, inplace=True)
    return df.loc[:, SELECT_COLUMNS]


def process_measurement_data(df: pd.DataFrame) -> pd.DataFrame:
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
