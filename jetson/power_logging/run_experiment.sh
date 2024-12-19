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
# NOTE : fcn_resnet50 is a object detection model and does not work with TorchTensorRT library
# TODO: Revisit fcn_resnet50 once this issue is addressed: https://github.com/pytorch/TensorRT/issues/3295
models=("alexnet" "vgg11" "vgg13" "vgg16" "vgg19" "mobilenet_v2" "mobilenet_v3_small" "mobilenet_v3_large" "resnet18" "resnet34" "resnet50" "resnet101" "resnet152" "lenet" "resnext50_32x4d" "resnext101_32x8d" "resnext101_64x4d" "convnext_tiny" "convnext_small" "convnext_base")
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
    --result-dir "$RESULT_DIR" \
    --optimization-level 3 \
    $INPUT_SHAPE
done

echo "Experiment completed!"
