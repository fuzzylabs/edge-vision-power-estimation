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

This is the most tricky part of the pipeline. We receive a raw dataset with time-stamped power logs and time-stamped layer runtimes as separate files. To obtain our training data, we have to find the power value that is closest to the time when that particular layer completed.

> [!NOTE]  
> This logic is implemented inside the `compute_layer_metrics_by_cycle` function of the [data_preprocess.py](./data_preparation/data_preprocess.py) python script.
