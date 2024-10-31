# TensorRT

Here I will provide an introduction and information on TensorRT library.

> NVIDIA® TensorRT™ is an SDK for high-performance deep learning inference on NVIDIA GPUs. It focuses on running an already-trained network quickly and efficiently on NVIDIA hardware.

TensorRT provides a good in-depth documentation : <https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html>.

## How it works?

There are two phases : Build phase and Inference phase as part of TensorRT model lifecycle.

For given input model (usually ONNX model)

### Build phase

This phase is responsible for building and producing an optimizing TensorRT engine.

Building a TensorRT engine consists of following steps

1. Create a network definition.

    The network is usually constructed by parsing an ONNX model. TensorRT also provides other approaches like creating a network layer by layer using using TensorRT’s `Layer` ([C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_layer.html), [Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/LayerBase.html#ilayer)) and `Tensor` ([C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_tensor.html), [Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/LayerBase.html#itensor)) interfaces. An example of creating network definition from scratch [here](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#create_network_python).

    In our code [build_engine.py](../trt/build_engine.py), this is done by `create_network` function. It creates a TensorRT network definition by parsing input ONNX model.

2. Specify a configuration for the builder.

    The `BuilderConfig` interface ([C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_builder_config.html), [Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/BuilderConfig.html)) is used to specify how TensorRT should optimize the mode.

    TensorRT supports FP32, FP16, BF16, FP8, INT4, INT8, INT32, INT64, UINT8, and BOOL data types. These depend on underlying hardware capabilities. Not all hardware support these data types. Not all operators support these data types as well. Refer to `Data Types` section for each the TensorRT Operator in the [documentation](https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/index.html) for details.

    Some interesting configuration
    * `builder_optimization_level` :  The builder optimization level which TensorRT should build the engine at. Setting a higher optimization level allows TensorRT to spend longer engine building time searching for more optimization options.
    * `max_num_tactics` :  The maximum number of tactics to time when there is a choice of tactics. Setting a larger number allows TensorRT to spend longer engine building time searching for more optimization options.
    * `add_optimization_profile` : By default, TensorRT optimizes the model based on the input shapes (batch size, image size, and so on) at which it was defined. However, the builder can be configured to adjust the input dimensions at runtime. TensorRT creates an optimized engine for each profile, choosing CUDA kernels that work for all shapes within the [minimum, maximum] range and are fastest for the optimization point - typically different kernels for each profile. You can then select among profiles at runtime.
    * `set_tactic_sources`: Various combinations of tactics that can be used by TensorRT while selecting optimal kernel. [Here is list](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/BuilderConfig.html#tensorrt.TacticSource) of all available tactics.

    In our code [build_engine.py](../trt/build_engine.py), this configuration is set by `self.config` instance variable.

3. Call builder to create a optimized TensorRT engine.

    Now we are set to build a TensorRT engine using network definition and builder configuration.

    The builder eliminates dead computations, folds constants, and reorders and combines operations to run more efficiently on the GPU. It can optionally reduce the precision of floating-point computations, either by simply running them in 16-bit floating point, or by quantizing floating point values so that calculations can be performed using 8-bit integers. It also times multiple implementations of each layer with varying data formats, then computes an optimal schedule to execute the model, minimizing the combined cost of kernel executions and format transforms.

    In our code [build_engine.py](../trt/create_engine.py), this is done by `create_engine` function. It creates a TensorRT network definition by parsing input ONNX model. The builder creates the engine in a serialized form called a plan, which can be deserialized immediately or saved to disk for later use.

### Inference phase

In inference phase, we use the optimized engine to run the inference. The rough steps involved in creating a inference runtime for this process are the following:

* Deserialize a plan to create an engine. (`engine` instance variable)
* Create an execution context from the engine. (`context` instance variable)
* For each input, we allocate input buffer and run inference using execution context. (`infer` method). It takes care of allocating space for input and output, moving input from host to GPU, and moving the output from GPU to host back.

In our code [infer.py](../trt/infer.py), creates an inference runtime and implements above steps.

**Notes**

* TensorRT has graph fusion optimizations, one engine layer may correspond to multiple ONNX ops in the original model. A list of various types of supported fusion: <https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#enable-fusion>.

* Serialized engines are not portable across platforms. Engines are specific to the exact GPU model that they were built on (in addition to the platform). The sections on [version compatibility](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#version-compat) and [hardware compatibility](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#hardware-compat) with some limitations.

* TensorRT builder uses timing to find the fastest kernel to implement a given layer. Timing kernels are subject to noise, such as other work running on the GPU, GPU clock speed fluctuations, etc. Timing noise means that the same implementation may not be selected on successive runs of the builder.

* [Timing Cache](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#timing-cache) records the latencies of each tactic for a specific layer configuration. The tactic latencies are reused if TensorRT encounters another layer with an identical configuration. Therefore, by reusing the same timing cache across multiple engine buildings runs with the same INetworkDefinition and builder config, you can make TensorRT select an identical set of tactics in the resulting engines

* If GPU clock speeds differ between engine serialization and runtime systems, the tactics chosen by the serialization system may not be optimal for the runtime system and may incur some performance degradation.

* [Algorithm selection and reproducible builds](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#algorithm-select) section provides insights on how to use the algorithm selector to achieve determinism and reproducibility in the builder. Picking the fastest algorithm in `selectAlgorithms` may not produce the best performance for the overall network, as it may increase reformatting overhead. The timing of an `IAlgorithm` is 0 in `selectAlgorithms` if TensorRT found that layer to be a no-op.

* Below are the descriptions about each builder optimization level:

  * Level 0: This enables the fastest compilation by disabling dynamic kernel generation and selecting the first tactic that succeeds in execution. This will also not respect a timing cache.

  * Level 1: Available tactics are sorted by heuristics, but only the top are tested to select the best. If a dynamic kernel is generated its compile optimization is low.

  * Level 2: Available tactics are sorted by heuristics, but only the fastest tactics are tested to select the best.

  * Level 3: Apply heuristics to see if a static precompiled kernel is applicable or if a new one has to be compiled dynamically.

  * Level 4: Always compiles a dynamic kernel.

  * Level 5: Always compiles a dynamic kernel and compares it to static kernels.

## Advanced

### API

TensorRT API provides C++ and Python language bindings.

> Note: The Python API is not available for all platforms. For more information, refer to the NVIDIA TensorRT Support Matrix.

**Profiling Layers**

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

        self.layers[layer_name].append(ms)
```

We can modify this `CustomProfiler` class to get timestamp as well when the layer executed (not exactly but with slight delay).

We attach this profiler using `enable_profiling` method of `TensorRTInfer` class in [infer.py](../trt/infer.py).

### Plugins

TensorRT has a `Plugin` interface that allows applications to provide implementations of operations that TensorRT does not support natively.

TensorRT ships with a library of plugins; the source for many of these and some additional plugins can be found [here](https://github.com/NVIDIA/TensorRT/tree/main/plugin).

### Best Practices for TensorRT performance

Guide: <https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#measure-performance>

Recommendation for best practices when running benchmarking or maximising performance using TensorRT SDK. It provides details on how to use CUDA profiling tools such as [NVIDIA Nsight™ Systems](https://developer.nvidia.com/nsight-systems).

To run our benchmark script with NVIDIA Nsight™ Systems, we run the following command

```bash
nsys profile -o alexnet-profile uv run benchmark.py
```

Next, open the generated `alexnet-profile.nsys-rep` file in the Nsight Systems GUI to visualize the captured profiling results.
