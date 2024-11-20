from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MICROWATTS_IN_WATTS = 1e6


@dataclass
class Dataset:
    """Base dataset."""

    csv_paths: list[Path]
    input_features: pd.DataFrame
    power: pd.Series
    runtime: pd.Series


@dataclass
class TrainTestDataset:
    """Train and Test Datasets."""

    train: Dataset
    test: Dataset


class DatasetBuilder:
    def __init__(self, features: list[str]) -> None:
        self.features = features

    @property
    def features_mapping(self) -> dict[str, int]:
        """Turn a list of features into a mapping from feature names to index."""
        return {feature: i for i, feature in enumerate(self.features)}

    def read_csv_and_convert_power(self, file_path: Path) -> pd.DataFrame:
        """Create a pandas dataframe from input CSV file.

        It converts power from microwatt to watt.

        Args:
            file_path: Path to CSV file.

        Returns:
            Dataframe where power column is converted from
            microwatts to watts
        """
        df = pd.read_csv(file_path)
        # Convert microwatt to watt
        df.power = df.power / MICROWATTS_IN_WATTS
        return df

    def merge_feature_data(self, file_paths: list[Path]) -> Dataset:
        """Read a list of path to CSV files and merge into 1 dataframe.

        Args:
            file_path: A list of paths to the CSV data file.
            features: List of feature column names.

        Returns:
            Dataset dataclass that contains input features,
            power and runtime values.
        """
        data = []
        for file_path in file_paths:
            df = self.read_csv_and_convert_power(file_path=file_path)
            data.append(df)

        df = pd.concat(data)
        input_features = df.loc[:, self.features]
        # Converted to watt from microwatt
        power = df.power
        # Recorded in milliseconds
        runtime = df.runtime
        return Dataset(
            csv_paths=file_paths,
            input_features=input_features,
            power=power,
            runtime=runtime,
        )

    def train_test_split(
        self, data_dir: Path, test_models: list[str], pattern: str
    ) -> tuple[list[Path] | None, list[Path] | None]:
        """Split dataset into train and test sets.

        Args:
            data_dir: Path to training dataset
            test_models: List of models to use as test set
            pattern: Pattern to find relevant CSV data files.

        Returns:
            Tuple of train and test sets.
            None is returned if pattern is not able to find
            any files.
        """
        csv_paths = list(data_dir.rglob(pattern))
        # Return None if there are no files matching the pattern
        if not len(csv_paths):
            return None, None

        train_paths, test_paths = [], []
        for file in csv_paths:
            # Get name of model from path
            model_name = file.parent.stem
            if model_name not in test_models:
                train_paths.append(file)
            else:
                test_paths.append(file)
        return train_paths, test_paths

    def create_dataset(
        self, data_dir: Path, test_models: list[str], pattern: str
    ) -> TrainTestDataset | None:
        """Create data for a convolutional layer CSVs.

        Args:
            data_dir: Path to training dataset
            test_models: List of models to use as test set
            pattern: Pattern to find relevant CSV data files.

        Returns:
            TrainTestDataset dataclass that contains
            training and testing datasets.
        """
        train_paths, test_paths = self.train_test_split(
            data_dir=data_dir, test_models=test_models, pattern=pattern
        )
        if train_paths is None or test_paths is None:
            return None
        train_dataset = self.merge_feature_data(train_paths)
        test_dataset = self.merge_feature_data(test_paths)
        return TrainTestDataset(train=train_dataset, test=test_dataset)
