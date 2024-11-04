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

### Training Data 

To model power consumption using the NeuralPower method, we need training data that captures energy usage per layer. For this, we follow the approach outlined in the paper [Profiling Energy Consumption of Deep Neural Networks on NVIDIA Jetson Nano](https://publik.tuwien.ac.at/files/publik_293778.pdf).