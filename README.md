# Power Consumption Prediction

This project contains script to measure and estimate power consumption of ML models on Jetson device.

## Structure

```bash
.
├── desktop         # Runtime benchmarking script for desktop
├── jetson          # Power and runtime measurement benchmarking script for jetson
├── model_training  # Power consumption modelling package
└── README.md

3 directories, 1 file
```

## Workflow

1. Run the benchmark using [jetson/power_logging](./jetson/power_logging/) scripts.
2. The benchmark data will be versioned (TODO).
3. Get the versioned data into [model_training](./model_training/) folder.
4. Run pre-processing scripts [map_power_to_layers.py](./model_training/map_power_to_layers.py) and [convert_measurements.py](./model_training/convert_measurements.py) to prepare training dataset.
5. Use the training dataset to train a model.

## Modelling Power Consumption

We use a method adapted from the [NeuralPower](https://arxiv.org/abs/1710.05420) paper.

### Training Data

To model power consumption using the NeuralPower method, we need training data that captures energy usage per layer. For this, we follow the approach outlined in the paper [Profiling Energy Consumption of Deep Neural Networks on NVIDIA Jetson Nano](https://publik.tuwien.ac.at/files/publik_293778.pdf).
