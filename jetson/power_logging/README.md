# Jetson

Measure power consumption and runtime for CNN models on the jetson device.

## 🔗 Quick Links

1. [Approach](#-approach)
2. [Repository Structure](#-repository-structure)
3. [Getting Started](#-getting-started)
4. [How it works?](#-how-it-works)
5. [Extras](#️-extras)

## 💡 Approach

The following process outlines the approach taken to collect the power and runtime values for each layer.

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

[Jetson Nano Orion Development Kit](https://developer.nvidia.com/embedded/learn/jetson-agx-orin-devkit-user-guide/index.html) - To run benchmarking experiments on Jetson board for collecting power and runtime measurements for a CNN model.

Following is the configuration of software and tools on the Jetson device used for testing:

```txt
JetPack 6.1
Jetson Linux 36.4
Docker 27.3.1
OS - Ubuntu 22.04-based root file system
```

[DagsHub account](https://dagshub.com/) and a repository for data versioning.

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

This will start running the [run_experiment.sh](./run_experiment.sh) script by default. You can also override by passing your custom experiment script. More information on the `run_experiment.sh` can be found in the [Experiment script](#experiment-script) section.

To follow the logs of the experiment, you can run the following command

```bash
sudo docker logs -f <container-name>
```

You can find the name of the docker container using the `sudo docker ps` command.

---

## ❓ How it works?

### Power measurement and logging

Jetson Orion Development Kit comes with three-channel INA3221 power monitor. The values of these modules can be read using `sysfs` nodes. The sys-file provides power, voltage and current measurements for the sensor under `i2c` folder.

> [!NOTE]  
> We read the power, voltage and current measurement for the sensor from the sys-file at this path `/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/in1_input` for Jetson Orion Development Kit.

These measurement values are saved in a separate file along with the timestamp when these values were read.

### TensorRT runtime profiling

We use [IProfiler](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Profiler.html) class to create a custom Profiler class.

```python
class CustomProfiler(trt.IProfiler):
    """Custom Profiler for logging layer-wise latency."""

    def __init__(self):
        trt.IProfiler.__init__(self)
        self.layers = {}

    def report_layer_time(self, layer_name, ms):
        if layer_name not in self.layers:
            self.layers[layer_name] = []

        # Time with seconds and microseconds
        current_time = datetime.now().strftime("%Y%m%d-%H:%M:%S.%f")
        self.layers[layer_name].append((ms, current_time))

```

We attach this profiler using `enable_profiling` method in [benchmark.py](./model/benchmark.py#L126). For each layer, it will record two values:

(i) Latency in milliseconds - Runtime

(ii) The timestamp when this value was recorded.

---

## 🗣️ Extras

### Experiment script

The [run_experiment.sh](./run_experiment.sh) script follows the approach outlined to collect power and runtime measurement for set of CNN models.

1. Lines 5-15: Measure idle power consumption for 120 seconds using `measure_idling_power.py` script.
2. Lines 16-18: Sleep for 120 seconds.
3. Lines 20-48: Benchmark the CNN models using [measure_inference_power.py](measure_inference_power.py) script.
4. Lines 50-62: Push the benchmark data to [fuzzylabs/edge-vision-power-estimation](https://dagshub.com/fuzzylabs/edge-vision-power-estimation) DagsHub repo for data version control.

### Raw Dataset Format

After container runs successfully, a `raw_data` folder is created and uploaded to DagsHub repository. Raw dataset is saved in the following format.

```bash
raw_data
├── alexnet
│   ├── alexnet_power_log.log
│   ├── alexnet_tensorrt.json
│   └── trt_profiling
│       ├── trt_engine_info.json
│       └── trt_layer_latency.json
....
├── idling_power.json
├── lenet
│   ├── lenet_power_log.log
│   ├── lenet_tensorrt.json
│   └── trt_profiling
│       ├── trt_engine_info.json
│       └── trt_layer_latency.json
....
```

It contains `idling_power.json` that records average idle power and a folder for each model recording the benchmarking values for the model.

#### Idle power

The average idle power is calculated and stored in `idling_power.json` file. An example of saved file is shown below. It saves two values: timestamp when idle power measurement was completed and the calculated average power in microwatts.

```json
{
    "timestamp": "20241121-051743",
    "avg_idle_power": 4234560.155322008
}
```

#### Model

The values for each model are recorded in a separate folder using model name as the folder name. There are 4 files saved as part of benchmarking the model.

`<model_name>_tensorrt.json`: This JSON file captures the overall latency taken for each inference cycle along with  the configuration used for benchmarking. The configuration parameters include input shape, data type of the input, number of inference cycles, number of warmup iterations, path to data directory. It also stores the average latency and average throughput. An example of saved file is shown below.

```json
{
    "config": {
        "model": "alexnet",
        "dtype": "float16",
        "input_shape": [
            1,
            3,
            224,
            224
        ],
        "warmup": 50,
        "runs": 30000,
        "optimization_level": 3,
        "min_block_size": 5,
        "result_dir": "raw_data"
    },
    "total_time": 263.73896875,
    ...
    "avg_latency": 0.00309211089823246,
    "avg_throughput": 323.40366594601403
}
```

`<model_name>_power_log.log`: This log file records the 3 values: timestamp, voltage and current measurements while the benchmarking experiment is being run in a separate process. An example of saved power log file is shown below

```txt
20241121-05:20:04.261448,5000.0,1096.0
20241121-05:20:04.262827,5000.0,1096.0
20241121-05:20:04.264174,5000.0,1096.0
20241121-05:20:04.265495,5000.0,1096.0
....
```

Inside `trt_profiling` folder for each model, we have two files `trt_engine_info.json` and `trt_layer_latency.json`.

`trt_engine_info.json`: This json file saves all the information for each layer in the TensorRT model. It saves detailed information such as padding, strides, dilation for a convolutional layer. An example of first layer in the Alexnet TensorRT model looks like following,

```json
{
    "Layers": [
        {
            "Name": "Reformatting CopyNode for Input Tensor 0 to [CONVOLUTION]-[aten_ops.convolution.default]-[/features/0/convolution] + [RELU]-[aten_ops.relu.default]-[/features/1/relu]",
            "LayerType": "Reformat",
            "Inputs": [
                {
                    "Name": "x",
                    "Location": "Device",
                    "Dimensions": [
                        1,
                        3,
                        224,
                        224
                    ],
                    "Format/Datatype": "Row major linear FP16 format"
                }
            ],
            "Outputs": [
                {
                    "Name": "Reformatted Input Tensor 0 to [CONVOLUTION]-[aten_ops.convolution.default]-[/features/0/convolution] + [RELU]-[aten_ops.relu.default]-[/features/1/relu]",
                    "Location": "Device",
                    "Dimensions": [
                        1,
                        3,
                        224,
                        224
                    ],
                    "Format/Datatype": "Channel major FP16 format where channel % 4 == 0"
                }
            ],
            "ParameterType": "Reformat",
            "Origin": "REFORMAT",
            "TacticValue": "0x00000000000003ea",
            "StreamId": 0,
            "Metadata": ""
        },
        ...
    ]
}
```

`trt_layer_latency.json`: This JSON file saves a dictionary mapping layer name to a tuple of (timestamp, latency) for each iteration of benchmarking. An example of how this file for first layer of Alexnet TensorRT model is shown below

```json
{
    "Reformatting CopyNode for Input Tensor 0 to [CONVOLUTION]-[aten_ops.convolution.default]-[/features/0/convolution] + [RELU]-[aten_ops.relu.default]-[/features/1/relu]": [
        [0.03030399978160858, "20241121-05:22:50.156286"],
        [0.10937599837779999, "20241121-05:22:50.159908"],
        ...
    ],
```

> [!NOTE]
> The runtime latency values are recorded in milliseconds. </br>
> The power consumption values are recorded in microwatts.

This raw data will be preprocessed and converted to a training dataset by scripts in [model_training](../../model_training/README.md) folder.
