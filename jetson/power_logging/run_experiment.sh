#!/bin/bash

set -eou pipefail

# Time (in seconds) to measure power in idle state
IDLE_DURATION=120

# Directory to store results
RESULT_DIR="raw_data/prebuilt_models"

echo "Running idling power measurement..."
python measure_idling_power.py \
  --idle-duration $IDLE_DURATION \
  --result-dir "$RESULT_DIR"

# Wait for 2 minutes
echo "Sleeping for 2 minutes..."
sleep 120

# Models to benchmark
# Using all YOLOv5 variants
models=("yolov5nu.pt" "yolov5su.pt" "yolov5mu.pt" "yolov5lu.pt" )

# Iterate through models and run measure_inference_power.py script
for model in "${models[@]}"
do
  echo "Running inference power measurement for model: $model"

  # Run the measure_inference_power.py script
  python measure_inference_power.py \
    --model "$model" \
    --dataset-name "coco.yaml" \
    --result-dir "$RESULT_DIR"
done

echo "Experiment completed!"
