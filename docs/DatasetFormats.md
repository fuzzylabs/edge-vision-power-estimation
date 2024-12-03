# Dataset Formats

In this document, we explain the dataset format for following datasets stored on DagsHub repository.

- [Raw Dataset](#raw-dataset-format)
- [Preprocessed Dataset](#preprocessing-dataset-format)
- [Training Dataset](#training-dataset-format)

> [!IMPORTANT]  
> DagsHub repository: <https://dagshub.com/fuzzylabs/edge-vision-power-estimation>

## Raw Dataset Format

A `raw_data` folder is created and uploaded to DagsHub repository as part of running the [Jetson power logging workflow](../jetson/power_logging/). Raw dataset is saved in the following format.

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

### Idle power

The average idle power is calculated and stored in `idling_power.json` file. An example of saved file is shown below. It saves two values: timestamp when idle power measurement was completed and the calculated average power in microwatts.

```json
{
    "timestamp": "20241121-051743",
    "avg_idle_power": 4234560.155322008
}
```

### Model

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
}
```

> [!NOTE]
> The runtime latency values are recorded in **milliseconds**. </br>
> The power consumption values are recorded in **microwatts**.

This raw data will be preprocessed and converted to a training dataset by scripts in [`model_training`](../model_training/) folder.

## Preprocessing Dataset Format

A `preprocessed_data` folder is created after running the `map_power_to_layers.py` script. It takes input the raw data folder, maps power and runtime values for each layer and creates a CSV.

Each model in `preprocessed_data` folder contains 2 files: `power_runtime_mapping_layerwise.csv` and `trt_engine_info.json`.

`trt_engine_info.json`: This is same JSON file that is created in the raw data. It contains detailed information such as padding, strides, dilation for a convolutional layer.

`power_runtime_mapping_layerwise.csv`: This CSV file contains per-layer data for each iteration of inference cycle. It includes information about current inference iteration, layer name, layer type, power consumed by the layer, runtime latency for the layer. An example entry of CSV for Alexnet model is shown below.

```csv
cycle,layer_name,layer_type,layer_power_including_idle_power_micro_watt,layer_power_excluding_idle_power_micro_watt,layer_run_time
1,Reformatting CopyNode for Input Tensor 0 to [CONVOLUTION]-[aten_ops.convolution.default]-[/feat_conv1/convolution] + [RELU]-[aten_ops.relu.default]-[/feat/relu],Reformat,5437392.769097799,1202832.6137757916,0.10390400141477585
```

## Training Dataset Format

A `training_data` folder is created after running `convert_measurements.py` script. It takes input the preprocessed data folder and splits the data into 3 CSV according to layer types of interest.

Each model in `training_data` folder contains 3 CSV files: `dense.csv`, `convolutional.csv` and `pooling.csv`.

Each CSV file stores relevant information such as batch_size, input_size, output_size, input_shape, output_shape, strides, padding, dilation, depending on the layer type.

An example of columns in Alexnet dense CSV is shown below

```csv
batch_size,input_size,output_size,power,runtime,layer_name
```

An example of columns in Alexnet pooling CSV is shown below

```csv
batch_size,input_size_0,input_size_1,input_size_2,output_size_0,output_size_1,output_size_2,kernel_0,kernel_1,stride_0,stride_1,power,runtime,layer_name
```

An example of columns in Alexnet convolutional CSV is shown below

```csv
batch_size,input_size_0,input_size_1,input_size_2,output_size_0,output_size_1,output_size_2,kernel_0,kernel_1,padding_0,padding_1,stride_0,stride_1,power,runtime,layer_name
```

Pooling and Convoluion store contains almost the same features except for padding in pooling layers. Dense layers store only input_size, output_size and batch_size information. All the layers in CSV contain values for power and runtime which is used for training the prediction models and evaluating the trained models.
