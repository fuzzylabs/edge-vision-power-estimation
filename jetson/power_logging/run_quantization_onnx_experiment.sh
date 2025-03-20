#!/bin/bash

set -eou pipefail

# Time (in seconds) to measure power in idle state
IDLE_DURATION=120

# Directory to store results
RESULT_DIR="raw_data/onnx_quantize_models"

echo "Running idling power measurement..."
python measure_idling_power.py \
  --idle-duration $IDLE_DURATION \
  --result-dir "$RESULT_DIR"

# Wait for 2 minutes
echo "Sleeping for 2 minutes..."
sleep 120

# Models to benchmark
# Using all YOLOv5 variants
# Note: Follow model_quantization/Readme.md to create ONNX and TensorRT models
models=("yolov5nu.onnx" "yolov5nu_quant.onnx" "yolov5su.onnx" "yolov5su_quant.onnx" "yolov5mu.onnx" "yolov5mu_quant.onnx" "yolov5lu.onnx" "yolov5lu_quant.onnx")

# Iterate through models and run measure_inference_power.py script
for model in "${models[@]}"
do
  echo "Running inference power measurement for model: $model"

  # Run the measure_inference_power.py script
  python measure_inference_power.py \
    --result-dir "$RESULT_DIR" \
    detect \
    --model "$model" \
    --dataset-name "coco.yaml"
done

echo "Experiment completed!"
