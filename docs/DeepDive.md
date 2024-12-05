# Behind the scenes

In this document, we shed some light on our process behind the scenes

## Jetson

### Power measurement and logging

The Jetson Orion Development Kit comes with a three-channel INA3221 power monitor. The values of these modules can be read using `sysfs` nodes. The sys-file provides power, voltage and current measurements for the sensor under the `i2c` folder.

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

We attach this profiler using `enable_profiling` method in [benchmark.py](./model/benchmark.py#L126). For each layer, it will record two values: latency in milliseconds - runtime taken and the timestamp when this value was recorded.

## Model Training

### Mapping power to layer runtimes

The [raw dataset](./DatasetFormats.md#raw-dataset-format) contains time-stamped power logs and time-stamped layer runtimes as separate files. To obtain the training data, we have to find the power consumed and runtime taken for a particular layer.

> [!NOTE]  
> This logic is implemented inside the `compute_layer_metrics_by_cycle` and `compute_latency_start_end_times` functions of the [data_preprocess.py](../model_training/data_preparation/data_preprocess.py) python script.

First we get the execution start time for the layer using the [`compute_latency_start_end_times`](../model_training/data_preparation/data_preprocess.py#L101) function. Here is am example of the latency data corresponding to the first layer of Alexnet raw data in [`trt_layer_latency.json`](https://dagshub.com/fuzzylabs/edge-vision-power-estimation/src/main/raw_data/alexnet/trt_profiling/trt_layer_latency.json) file.

```json
{
    "Reformatting CopyNode for Input Tensor 0 to [CONVOLUTION]-[aten_ops.convolution.default]-[/features/0/convolution] + [RELU]-[aten_ops.relu.default]-[/features/1/relu]": [
        [0.03030399978160858, "20241121-05:22:50.156286"],
        [0.10937599837779999, "20241121-05:22:50.159908"],
        ...
    ],
}
```

The start time of the first iteration cycle containing a list of `[latency_in_ms, layer_executed_timestamp]` values as `[0.03030399978160858, "20241121-05:22:50.156286"]` is calculated as `layer_executed_timestamp-latency_in_ms`. This operation gives the start time for the first iteration for the first layer as `2024-11-21 05:22:50.156256`.

Once we have `(start_timestamp, end_timestamp, execution_duration)` for every layer for all the iteration cycles, we get the power consumed by the layer during the inference.

To get the power values, we use the power logs file.  [`alexnet_power_log.log`](https://dagshub.com/fuzzylabs/edge-vision-power-estimation/src/main/raw_data/alexnet/alexnet_power_log.log) file. This file contains 3 values on each row : timestamp, voltage, and current. These values are recorded while the benchmarking experiment is being run.

```txt
20241121-05:20:04.261448,5000.0,1096.0
20241121-05:20:04.262827,5000.0,1096.0
20241121-05:20:04.264174,5000.0,1096.0
20241121-05:20:04.265495,5000.0,1096.0
....
```

We have to find all the power values (voltage*current) in between `start_timestamp` and `end_timestamp` for each iteration cycle for all the layers. The power consumed for a layer is the average of all the power values between the `start_timestamp` and the `end_timestamp`.
