# Power Consumption Prediction

This project contains script to measure and estimate power consumption of ML models on Jetson device.

## Structure

```bash
.
├── desktop   # Power measurement benchmarking script for desktop
├── jetson    # Power measurement benchmarking script for jetson
├── model     # Power consumption modelling package
└── README.md

2 directories, 1 file
```

## Modeling Power Consumption

We use a method adapted from the [NeuralPower](https://arxiv.org/abs/1710.05420) paper.

