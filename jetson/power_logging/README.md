# Jetson

Measure power consumption and runtime for CNN models on the jetson device.

## 🔗 Quick Links

1. [Approach](#-approach)
2. [Repository Structure](#-repository-structure)
3. [Getting Started](#-getting-started)
4. [How it works?](#-how-it-works)
5. [Extras](#️-extras)

## 💡 Approach

To collect the power and runtime values for each layer,

First, we measure the idle power value of the Jetson. This power value measures how much power is consumed when minimal required processes are running on the Jetson.

> [!CAUTION]
> The recommendation is to disable any GUI operations and use command line interface on Jetson  to reduce the number of background processes for getting the idle power.

Next, we run two separate process on Jetson wherein the first process runs the benchmarking for a CNN model. This process captures the per-layer runtime for the model. In the second process, we launch the power logging script. Two separate processes are used to ensure that the benchmarking and power logging tasks are performed concurrently without interference. This approach prevents the benchmarking process from being slowed down by the additional overhead of logging power measurements.

Finally, we upload the collection of power and runtime data for each model to DagsHub. This is the raw data that we will further preprocess to create training data. This dataset is versioned using DVC.

---

## 📂 Repository Structure

```bash
.
├── assets
├── data_version.py
├── Dockerfile.jetson
├── docs
├── measure_idling_power.py
├── measure_inference_power.py
├── measure_power.py
├── model                       # Benchmarking utility functions
├── pyproject.toml
├── README.md
├── run_experiment.sh
└── uv.lock
```

- **[data_version.py](./data_version.py)** : This script contains functions to upload and download dataset to/from Dagshub. DagsHub uses DVC underneath to create data versions.

- **[measure_idling_power.py](./measure_idling_power.py)** : This script measures average power usage when there Jetson is idle i.e. no benchmarking is being run.

- **[measure_power.py](./measure_power.py)** : This scripts provides a function to read power values from  INA3221 power monitor sensor on Jetson device.

- **[run_experiment.sh](./run_experiment.sh)** : Experiment script that runs the power and runtime collection process end-to-end.

---

## 🛸 Getting Started

### ⚙️ System and Hardware Requirements

- [Jetson Nano Orion Development Kit](https://developer.nvidia.com/embedded/learn/jetson-agx-orin-devkit-user-guide/index.html) - To run benchmarking experiments on Jetson board for collecting power and runtime measurements for a CNN model.

Following is the configuration of software and tools on the Jetson device used for testing:

```txt
JetPack 6.1
Jetson Linux 36.4
Docker 27.3.1
OS - Ubuntu 22.04-based root file system
```

### 💨 Run Experiment Script

1. To maximise the Jetson power and fan speed run the following command on Jetson.

    ```bash
    sudo nvpmodel -m 0
    sudo jetson_clocks
    ```

2. Build the docker image

    ```bash
    sudo docker build -t edge-vision-benchmark -f Dockerfile.jetson .
    ```

    > [!IMPORTANT]  
    > Use this exact Docker image to ensure compatibility with `tensorrt==10.1.0` and `torch_tensorrt==2.4.0`.<br>
    > Base image `nvcr.io/nvidia/pytorch:24.06-py3-igpu` might take some time to download. (approx. 5 GB in size)

3. Run the container

    ```bash
    sudo docker run -e DAGSHUB_USER_TOKEN=<dagshub-token> --runtime=nvidia --ipc=host -v $(pwd):/app -d edge-vision-benchmark
    ```

    > [!NOTE]  
    > You can generate a long lived app DagsHub token with no expiry date from your [User Settings](https://dagshub.com/user/settings/tokens).

This will start running the [run_experiment.sh](./run_experiment.sh) script by default. You can also override by passing your custom experiment script.

To follow the logs of the experiment, you can run the following command

```bash
sudo docker logs -f <container-name>
```

---

## ❓ How it works?

### Power logging and measurement

---

## 🗣️ Extras
