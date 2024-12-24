# Experiment Scripts

In this document, we provide a detailed breakdown of individual scripts used for experimentation.

- [Collect power and runtime measurements on Jetson](#data-collection-script-on-jetson)
- [Data preprocessing](#data-preprocessing-script)

## Data collection script on Jetson

The [`run_experiment.sh`](../jetson/power_logging/run_experiment.sh) script is responsible for collecting power and runtime measurements for CNN models on a Jetson device. The script performs the following operations:

1. `Lines 5-15`: Measure idle power consumption for 120 seconds using `measure_idling_power.py` script.
2. `Lines 16-18`: Sleep for 120 seconds.
3. `Lines 20-48`: Benchmark the CNN models using [measure_inference_power.py](measure_inference_power.py) script.

## Data Preprocessing script

The [`create_dataset.sh`](../model_training/create_dataset.sh) script provides a helpful utility to get preprocessed and training datasets from a raw dataset. The script performs the following operations:

1. `Lines 12:15`: Create preprocessed dataset from raw dataset using the `map_power_to_layers.py` script.
2. `Lines 20:23`: Create training dataset from preprocessed dataset using the `convert_measurements.py` script.
