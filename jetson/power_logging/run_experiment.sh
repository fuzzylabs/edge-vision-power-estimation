#!/bin/bash

set -eou pipefail

# Time (in seconds) to measure power in idle state
IDLE_DURATION=120

# Directory to store results
RESULT_DIR="raw_data/single_conv_layer_trt"

echo "Running idling power measurement..."
python measure_idling_power.py \
  --idle-duration $IDLE_DURATION \
  --result-dir "$RESULT_DIR"

# Wait for 2 minutes
echo "Sleeping for 2 minutes..."
sleep 120

# Models to benchmark
models=("conv_k1_s1_o3" "conv_k2_s1_o3" "conv_k3_s1_o3" "conv_k4_s1_o3" "conv_k5_s1_o3" "conv_k1_s2_o3" "conv_k1_s3_o3" "conv_k1_s4_o3" "conv_k1_s5_o3" "conv_k1_s1_o6" "conv_k1_s1_o12" "conv_k1_s1_o24" "conv_k1_s1_o48")
# Number of inference cycles
RUNS=3000

# Iterate through models and run measure_inference_power.py script
for model in "${models[@]}"
do
  echo "Running inference power measurement for model: $model"

INPUT_SHAPE='--input-shape 1 3 224 224'

  # Run the measure_inference_power.py script
  python measure_inference_power.py \
    --result-dir "$RESULT_DIR" \
    classify \
    --model "$model" \
    --runs "$RUNS" \
    --dtype "float32" \
    $INPUT_SHAPE
done

echo "Experiment completed!"
