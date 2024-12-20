#!/bin/bash

set -eou pipefail

# Local Directory to store raw data
RAW_DATA_DIR="../jetson/power_logging/raw_data"

# Local Directory to store preprocessed data
PREPROCESSED_DATA_DIR="preprocessed_data"

echo "Preprocess raw data"
python map_power_to_layers.py \
    --raw-data-dir "$RAW_DATA_DIR" \
    --result-dir "$PREPROCESSED_DATA_DIR"

# Local Directory to store training data
TRAIN_DATA_DIR="training_data"

echo "Prepare training data"
python convert_measurements.py \
    --preprocessed-data-dir "$PREPROCESSED_DATA_DIR" \
    --result-dir "$TRAIN_DATA_DIR"

echo "Experiment completed!"
