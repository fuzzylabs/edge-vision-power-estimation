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

# Default values for flags
PUSH_TO_DAGSHUB=false

# Parse flags whether to push data to DagsHub
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --push-to-dagshub) PUSH_TO_DAGSHUB=true ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

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


# Local Directory to store training data
TRAIN_DATA_DIR="training_data"

echo "Prepare training data"
python convert_measurements.py \
    --preprocessed-data-dir "$PREPROCESSED_DATA_DIR" \
    --result-dir "$TRAIN_DATA_DIR"


if [[ "$PUSH_TO_DAGSHUB" == true ]]; then
  COMMIT_MESSAGE="Add second version of training data"
  echo "Push training data to DagsHub"
  python data_version.py \
    --owner "$DAGSHUB_OWNER" \
    --name "$DAGSHUB_REPO_NAME" \
    --local-dir-path "$TRAIN_DATA_DIR" \
    --commit "$COMMIT_MESSAGE" \
    --upload
fi

echo "Experiment completed!"
