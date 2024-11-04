from pathlib import Path

import pandas as pd

columns = [
    "layer_name",
    "average_power",
    "average_run_time",
]


def read_measurement_data(path: Path) -> pd.DataFrame:
    """Read measurement data from CSV file."""
    df = pd.read_csv(path)
    df.rename(columns={"layer name": "layer_name"}, inplace=True)
    return df.loc[:, columns]
