#!/bin/bash

# Directory to store results
RESULT_DIR = "raw_data/prebuilt_models"

# List of models to process
MODELS = ("alexnet" "vgg11" "vgg13" "vgg16" "vgg19" "mobilenet_v2" "mobilenet_v3_small" "mobilenet_v3_large" "resnet18" "resnet34" "resnet50" "resnet101" "resnet152" "lenet" "resnext50_32x4d" "resnext101_32x8d" "resnext101_64x4d" "convnext_tiny" "convnext_small" "convnext_base")

PYTHON_SCRIPT = "save_model_summary.py"

for MODEL in "${MODELS[@]}"
do
    MODEL_OUTPUT_DIR="${RESULT_DIR}/${MODEL}"
    OUTPUT_FILE = "${MODEL_OUTPUT_DIR}/model_summary.json"

    mkdir -p "$MODEL_OUTPUT_DIR"

    echo "Generating summary for model: $MODEL"
    python "$PYTHON_SCRIPT" --model "$MODEL" --output-file "$OUTPUT_FILE"

    if [ $? -eq 0]; then
        echo "Summary saved to: $OUTPUT_FILE"
    else
        echo "Failed to generate summary for model: $MODEL"
    fi
done