"""Integration test for preprocessed and training data."""


def test_create_dataset():
    """Integration test.

    We create a test that tests our workflow for generating
    training dataset from raw dataset. There are two steps

    1. Using raw dataset, generate preprocessed dataset
    using `map_power_to_layers.py`
    2. Using preprocessed data, create training dataset
    using `convert_measurements.py`
    """
    pass
