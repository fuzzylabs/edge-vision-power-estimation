# EdgeProfiler
<p align="center">
<img src="./assets/intro.jpg" alt="readme-ai-banner-logo" width="80%">
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="license">
  </a>
</p>

## 🔗 Quick Links

* [Documentation](docs)
* Source Code
  * [Inference](model_training/inference.py)
  * [Training](model_training/run.py)
  * [Data Collection](jetson/power_logging/)
* [Data Hub](https://dagshub.com/fuzzylabs/edge-vision-power-estimation)
* [Experiment Tracking](https://dagshub.com/fuzzylabs/edge-vision-power-estimation.mlflow/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D)
* [Model Hub](https://dagshub.com/fuzzylabs/edge-vision-power-estimation.mlflow/#/models)

## 🔮 Introduction

**Measure smarter, deploy greener:** A tool for inferring and optimising the power consumption of Convolutional Neural Networks (CNN's) on edge devices.

EdgeProfiler helps you understand and minimise your power consumption and runtime without having to leave your development machine. This allows you to gauge your deployment environmental impact during or even before the training process to help you to make smarter training decisions. 

---

**What's inside:**
- **Inference:** Determine power consumption and runtime for different layers in a CNN model on an Nvidia Jetson edge device.
- **Training:** Build your own power consumption and runtime models using Scikit-Learn and MLFlow for experiment tracking.
- **Collection:** Record measurements of a model's power consumption and runtime during inference, storing all data versions in DagsHub.

<details>
	<summary>💡 Why this approach?</summary>

  The approach we take is similar to that of the <a href="https://arxiv.org/abs/1710.05420">NeuralPower paper</a>. We use the same methodology focusing on data collection, model training, and power prediction for TensorRT models on edge devices.

  > **Tip:** More information on NeuralPower is documented [here](./docs/NeuralPower.md) and TensorRT is documented [here](./desktop/torch_onnx_trt/docs/TensorRT.md).
</details>

<details>
  <summary>🎯 What have we done?</summary>

- We have collected power and runtime measurements on a Jetson Orion device for 21 models. The dataset can be found on the DagsHub repository.

- We have trained power and runtime prediction models for 3 different layer types of CNN models. The experiments can be viewed in the DagsHub MLFlow UI.

> 📌 **Important**
> 
> DagsHub repository: <https://dagshub.com/fuzzylabs/edge-vision-power-estimation> </br> </br>
> MLFlow UI: <https://dagshub.com/fuzzylabs/edge-vision-power-estimation.mlflow/>

Learn more about how to get started to train power and runtime prediction models in [Run Locally](#-run-locally) section.
</details>

<details>
<summary>📍 How did we get here?</summary>

This project is divided into two stages: [Experimental](#-experimental-stage) and [Implementation](#-implementation-stage)

### 🧪 Experimental Stage

In the experimental stage we experimented with different approaches for benchmarking the conversion of TensorRT models.

> **Tip:** All code and benchmarking scripts for the 3 experimental approaches can be found in the desktop folder: [README](./desktop/README.md)

### 🚀 Implementation Stage

In the implementation stage of the project we created an MLOps pipeline for training the power consumption and runtime prediction models. The pipeline consists of the following two step process:

📊 Jetson Device Benchmarking

1. Collect the power consumption and performance data on the Jetson device
2. Follow the detailed process outlined in [jetson/power_logging](./jetson/power_logging/README.md)

🤖 Machine Learning Model Development

1. Preprocess collected data from the Jetson device
2. Train machine learning models for power and runtime prediction
3. Refer to the comprehensive guide in [model_training](./model_training/README.md)

</details>

### 🏎💨 Have a go yourself

To get started, set up your python environment. We really like using `uv` for package and project management - so if you don't have it go ahead and follow their installation guide from [here](https://docs.astral.sh/uv/getting-started/installation/).

Once you have it installed run the following commands from inside the [`model_training`](/model_training) directory.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

<details open>
  <summary>🧠 Power Inference</summary>

Run the following inference script to predict the power consumption and runtime for  a `resnet18` model on a Jetson Nano device:

```commandline
python inference.py \
    --trt-engine-path sample_data/resnet18_trt_engine_info.json \
    --result-csv-path results/resnet18_predictions.csv
```

> **Tip:** For more details on running inference see the  [Inference](./model_training/README.md#-inference) section of the [`model_training`](model_training/README.md) README.

</details>

<details>
  <summary>🏋️ Model Training</summary>

  <details>
  <summary>Data Collection</summary>

  If you do have access to the Jetson device, feel free to follow the step by step guide outlined in the [getting started](./jetson/power_logging/README.md#-getting-started) section of the [`jetson/power_logging`](jetson/power_logging/README.md) README to collect your own measurements.
  
  If you don't have access to a Jetson device yourself, we recommend pulling our training data from DagsHub by following the step by step guide outlined in the [getting started](./model_training/README.md#-getting-started) section of the `model_training` README.

  </details>
  
  Once you have access to training data you can train your own model with our training script:

```commandline
python run.py
```

> **Tip:** For more details on training your own model see the  [Run Training Script](./model_training/README.md#-run-training-script) section of the [`model_training`](model_training/README.md) README.


</details>

## 🔰 Contributing

Contributions are welcome! Please read the [Contributing Guide](./CONTRIBUTING.mds) to get started.

- **💡 [Contributing Guide](./CONTRIBUTING.md)** : Learn about our contribution process and coding standards.
- **🐛 [Report an Issue](https://github.com/fuzzylabs/edge-vision-power-estimation/issues)** : Found a bug? Let us know!
- **💬 [Start a Discussion](https://github.com/fuzzylabs/edge-vision-power-estimation/discussions)** : Have ideas or suggestions? We'd love to hear from you.

## 🙌 Acknowledgements
The following resources have served as an inspiration for this project:
- [NeuralPower paper](https://arxiv.org/pdf/1710.05420) authors
- [Profiling Energy Consumption of Deep Neural Networks on NVIDIA Jetson Nano](https://publik.tuwien.ac.at/files/publik_293778.pdf) authors
- [readme-ai](https://github.com/eli64s/readme-ai) for Readme inspiration

## 📄 License

Copyright © 2024 [Fuzzy Labs](./README.md). <br />
Released under the [Apache 2.0](./LICENSE)
