# Power Consumption Prediction

This project contains script to measure and estimate power consumption of ML models on Jetson device.

## Structure

```bash
.
├── desktop   # Power measurement benchmarking script for desktop
├── jetson    # Power measurement benchmarking script for jetson
├── model_training  # Power consumption modelling package
└── README.md

3 directories, 1 file
```

## Modelling Power Consumption

We use a method adapted from the [NeuralPower](https://arxiv.org/abs/1710.05420) paper.

