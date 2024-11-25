#!/bin/bash

set -eou pipefail

DAGSHUB_OWNER="fuzzylabs"
DAGSHUB_REPO_NAME="edge-vision-power-estimation"
# or we can pass SHA commit as well to download state at that particular commit 
BRANCH_NAME="main" 

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
  --branch "$BRANCH_NAME" \
  --download


# Local Directory to store preprocessed data
PREPROCESSED_DATA_DIR="preprocessed_data"

echo "Preprocess raw data"
python map_power_to_layers.py \
    --raw-data-dir "$RAW_DATA_DIR" \
    --result-dir "$PREPROCESSED_DATA_DIR"

COMMIT_MESSAGE="Add second version of preprocessed data"

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
python convert_measurements.py \
    --preprocessed-data-dir "$PREPROCESSED_DATA_DIR" \
    --result-dir "$TRAIN_DATA_DIR"

COMMIT_MESSAGE="Add second version of training data"

echo "Push training data to DagsHub"
python data_version.py \
  --owner "$DAGSHUB_OWNER" \
  --name "$DAGSHUB_REPO_NAME" \
  --local-dir-path "$TRAIN_DATA_DIR" \
  --commit "$COMMIT_MESSAGE" \
  --upload

echo "Experiment completed!"
