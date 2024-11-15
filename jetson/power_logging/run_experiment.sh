#!/bin/bash

set -eou pipefail

# Time (in seconds) to measure power in idle state
IDLE_DURATION=120

# Directory to store results
RESULT_DIR="raw_data"

echo "Running idling power measurement..."
python measure_idling_power.py \
  --idle-duration $IDLE_DURATION \
  --result-dir "$RESULT_DIR"

# Wait for 2 minutes
echo "Sleeping for 2 minutes..."
sleep 120

# Models to benchmark
# Using models mentioned in Neural Power
# Section 3.3 covers various CNN architectures used for the experiment
models=("alexnet" "vgg16" "vgg19" "googlenet" "fcn_resnet50" "mobilnet_v2" "resnet18" "lenet")

# Number of inference cycles
RUNS=30000

# Iterate through models and run measure_inference_power.py script
for model in "${models[@]}"
do
  echo "Running inference power measurement for model: $model"

  # Set input shape to be different for lenet model
  if [ "$model" == "lenet" ]; then
    INPUT_SHAPE='--input-shape 1 1 32 32'
  else
    INPUT_SHAPE='--input-shape 1 3 224 224'
  fi

  # Run the measure_inference_power.py script
  python measure_inference_power.py \
    --model "$model" \
    --runs "$RUNS" \
    --optimization-level 3 \
    $INPUT_SHAPE
done

DAGSHUB_OWNER="fuzzylabs"
DAGSHUB_REPO_NAME="edge-vision-power-estimation"
COMMIT_MESSAGE="Add raw data"

echo "Using DagsHub and DVC for benchmark data"
python data_version.py \
  --owner "$DAGSHUB_OWNER" \
  --name "$DAGSHUB_REPO_NAME" \
  --local-dir-path "$RESULT_DIR" \
  --commit "$COMMIT_MESSAGE" \
  --upload

echo "Experiment completed!"
