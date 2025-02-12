#!/bin/bash

set -eou pipefail

# Time (in seconds) to measure power in idle state
IDLE_DURATION=120

# Directory to store results
RESULT_DIR="raw_data/single_conv_layer"

echo "Running idling power measurement..."
python measure_idling_power.py \
  --idle-duration $IDLE_DURATION \
  --result-dir "$RESULT_DIR"

# Wait for 2 minutes
echo "Sleeping for 2 minutes..."
sleep 120

# Models to benchmark
models=("single_conv_layer")
# Number of inference cycles
RUNS=3000

# Iterate through models and run measure_inference_power.py script
for model in "${models[@]}"
do
  echo "Running inference power measurement for model: $model"

INPUT_SHAPE='--input-shape 1 3 224 224'

  # Run the measure_inference_power.py script
  python measure_inference_power.py \
    classify \
    --model "$model" \
    --runs "$RUNS" \
    --dtype "float32" \
    --result-dir "$RESULT_DIR" \
    $INPUT_SHAPE
done

echo "Experiment completed!"
