#!/bin/bash

set -eou pipefail

DAGSHUB_OWNER="fuzzylabs"
DAGSHUB_REPO_NAME="edge-vision-power-estimation"
COMMIT_MESSAGE="Add raw data"

# Local Directory to store raw data
RAW_DATA_DIR="raw_data"

# Remote Directory to pull raw data
REMOTE_RAW_DATA_DIR="raw_data"

echo "Pull benchmark data from DagsHub"
python data_version.py \
  --owner "$DAGSHUB_OWNER" \
  --name "$DAGSHUB_REPO_NAME" \
  --local-dir-path "$RAW_DATA_DIR" \
  --remote-dir-path "$REMOTE_RAW_DATA_DIR" \
  --download


# Local Directory to store preprocessed data
PREPROCESSED_DATA_DIR="preprocessed_data"

echo "Preprocess raw data"
# python map_power_to_layers.py \
#     --idling-power-log-path results/60_seconds_idling_power_log_20241103-144950.log \
#     --raw-data-dir "$RAW_DATA_DIR" \
#     --result-dir "$PREPROCESSED_DATA_DIR"

COMMIT_MESSAGE="Add preprocessed data"
echo "Push preprocessed data to DagsHub"
python data_version.py \
  --owner "$DAGSHUB_OWNER" \
  --name "$DAGSHUB_REPO_NAME" \
  --local-dir-path "$PREPROCESSED_DATA_DIR" \
  --commit "$COMMIT_MESSAGE" \
  --upload


# Local Directory to store preprocessed data
TRAIN_DATA_DIR="training_data"


echo "Prepare training data"
# python convert_measurements.py \
#     "$TRAIN_DATA_DIR" \
#     "$PREPROCESSED_DATA_DIR"

COMMIT_MESSAGE="Add training data"

echo "Push training data to DagsHub"
python data_version.py \
  --owner "$DAGSHUB_OWNER" \
  --name "$DAGSHUB_REPO_NAME" \
  --local-dir-path "$TRAIN_DATA_DIR" \
  --commit "$COMMIT_MESSAGE" \
  --upload

echo "Experiment completed!"
