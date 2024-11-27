
<p align="center">
<img src="./assets/intro.jpg" alt="readme-ai-banner-logo" width="80%">>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="license">
  </a>
</p>

---

## 🔗 Quick Links

1. [Overview](#-overview)
2. [Repository Structure](#-repository-structure)
3. [Getting Started](#-getting-started)
4. [Contributing](#-contributing)
5. [License](#-license)

---

## 🔮 Overview

This project aims to reproduce [NeuralPower paper](https://arxiv.org/abs/1710.05420) for TensorRT models on edge devices, focusing on data collection, model training, and power prediction. It seeks to assess power consumption metrics for machine learning models while optimising energy efficiency, accuracy, and reliability.

> [!TIP]  
> More information on NeuralPower is documented [here](./docs/NeuralPower.md) and TensorRT is documented [here](./desktop/torch_onnx_trt/docs/TensorRT.md).

TL;DR: NeuralPower paper introduces a process to predict power and runtime consumption of CNN-based models. Their methodology used Caffe models. We use the same process as the paper but for TensorRT models. TensorRT models are optimized model inferencing engine for Nvidia GPUs and Jetson devices.

---

## 📂 Repository Structure

```bash
.
├── assets
├── CONTRIBUTING.md
├── desktop                       # Runtime benchmarking script for desktop
├── jetson
│   └── power_logging             # Power and runtime measurement benchmarking script for jetson
├── LICENSE
├── model_training                # Power and runtime prediction modelling package
└── README.md
```

- **[desktop](./desktop/README.md)** : This folder contains initial exploration of various approaches to get a TensorRT model from PyTorch model on a desktop/laptop.

- **[jetson/power_logging](./jetson/power_logging/README.md)** : Jetson folder contains scripts to collect power and runtime measurements, `raw_data` for a Convolutional Neural Network (CNN) model on Jetson devices. For this experiment, we used [Jetson Nano Orion Development Kit](https://developer.nvidia.com/embedded/learn/jetson-agx-orin-devkit-user-guide/index.html).

- **[model_training](./model_training/README.md)**: Model training folder uses the `raw_data` collected on the Jetson device to train power and runtime prediction models using sklearn.

---

## 🛸 Getting Started

### ⚙️ System and Hardware Requirements

- [Jetson Nano Orion Development Kit](https://developer.nvidia.com/embedded/learn/jetson-agx-orin-devkit-user-guide/index.html) - To run benchmarking experiments on Jetson board for collecting power and runtime measurements for a CNN model.

> [!NOTE]  
> If you do not have access to a Jetson device, you can use uploaded raw data from [DagsHub repository](https://dagshub.com/fuzzylabs/edge-vision-power-estimation) to get started. More information about these datasets can be found in the [Datasets](#-dataset) section.

For rest of the project following tools are used,

- [uv](https://docs.astral.sh/uv/) : It is used as default for running this project locally.

- (Optional) [Docker](https://docs.docker.com/get-started/) : It is used for running the experiments under [desktop](./desktop/) folders. `Docker` is as an alternative to `uv` run the different approaches to convert and benchmark PyTorch models to TensorRT models.

### 📍 Workflow

This project is divided into two stages:  Experimental and Implementation

#### 🧪 Experimental Stage

This part of the project focused on experimenting with different approaches for benchmarking various approaches to convert TensorRT models.

> [!TIP]  
> All code and benchmarking script for 3 experimental approaches can be found under desktop folder: [README](./desktop/README.md)

#### 🚀 Implementation Stage

This part of the project creates a MLOps pipeline. It is a two-step process

📊 Jetson Device Benchmarking

1. Collect the power consumption and performance data on the Jetson device
2. Follow the detailed process outlined in [jetson/power_logging](./jetson/power_logging/README.md)

🤖 Machine Learning Model Development

1. Preprocess collected data from the Jetson device
2. Generate and train machine learning models
3. Refer to the comprehensive guide in [model_training](./model_training/README.md)

### 💾 Dataset

[DagsHub and DVC integration](https://dagshub.com/docs/integration_guide/dvc/) is used for data versioning.The datasets are managed and versioned using DVC, enabling seamless tracking of changes and reproducibility across different stages of the project.

> [!IMPORTANT]  
> DagsHub repository: <https://dagshub.com/fuzzylabs/edge-vision-power-estimation>

Currently, there are two versions of datasets.

- [First version](https://dagshub.com/fuzzylabs/edge-vision-power-estimation/src/b35eb12d9c9be397f32d54f3fce6d1322862a8a0) : First version of the benchmarking dataset collected on the Jetson device consists of 7 models.

- [Second version](https://dagshub.com/fuzzylabs/edge-vision-power-estimation/src/cfd51e06079b7ab363b44db6a633fe74f5443022) : Second version of the benchmarking dataset adds 14 new models, totalling to a collection of 21 models dataset.

Each version of the dataset consists of 3 folders

- `raw_data` : Contains the unprocessed data directly collected from the Jetson device.
- `preprocessed_data` : Raw dataset is preprocessed to transformed and formatted to ensure consistency and usability.
- `training_data` : This folder contains the final dataset prepared for training machine learning models.

### 📊 Experiment Tracker

[DagsHub and MLflow integration](https://dagshub.com/docs/integration_guide/mlflow_tracking/) is used as experiment tracker.

> [!IMPORTANT]  
> MLFlow UI: <https://dagshub.com/fuzzylabs/edge-vision-power-estimation.mlflow/>

There are two experiments logged on the MLflow experiment tracker corresponding to each version of the dataset.

---

## 🔰 Contributing

Contributions are welcome! Please read the [Contributing Guide](./CONTRIBUTING.mds) to get started.

- **💡 [Contributing Guide](./CONTRIBUTING.md)** : Learn about our contribution process and coding standards.
- **🐛 [Report an Issue](https://github.com/fuzzylabs/edge-vision-power-estimation/issues)** : Found a bug? Let us know!
- **💬 [Start a Discussion](https://github.com/fuzzylabs/edge-vision-power-estimation/discussions)** : Have ideas or suggestions? We'd love to hear from you.

---

## 📄 License

Copyright © 2024 [Fuzzy Labs](./README.md). <br />
Released under the [Apache 2.0](./LICENSE)
