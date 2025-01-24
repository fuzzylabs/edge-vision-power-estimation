#!/bin/bash

set -eou pipefail

# Directory to save the outputs
OUTPUT_DIR="raw_data/prebuilt_models"

# List of models to process
MODELS=("alexnet" "vgg16" "vgg13" "vgg19" "mobilenet_v2" "mobilenet_v3_small" 
        "mobilenet_v3_large" "resnet18" "resnet34" "resnet50" "resnet101" 
        "resnet152" "lenet" "resnext50_32x4d" "resnext101_32x8d" 
        "resnext101_64x4d" "convnext_tiny" "convnext_small" "convnext_base")

# Python script to run
PYTHON_SCRIPT="./save_model_summary.py"

# Loop through each model
for MODEL in "${MODELS[@]}"
do
    # Define default input shape
    INPUT_SHAPE="1 3 224 224"

    # Adjust input shape for specific models
    if [ "$MODEL" == "lenet" ]; then
        INPUT_SHAPE="1 1 32 32"
    fi

    # Define the output path for this model
    MODEL_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL}"
    OUTPUT_FILE="${MODEL_OUTPUT_DIR}/model_summary.json"

    # Run the Python script and save the output
    echo "Generating summary for model: $MODEL with input shape: $INPUT_SHAPE"
    python "$PYTHON_SCRIPT" --model "$MODEL" --input-shape $INPUT_SHAPE --output-file "$OUTPUT_FILE"

    if [ $? -ne 0 ]; then
        echo "Skipping model: $MODEL due to loading error."
        continue
    fi

    if [ $? -eq 0 ]; then
        echo "Summary saved to: $OUTPUT_FILE"
    else
        echo "Failed to generate summary for model: $MODEL"
    fi

done

