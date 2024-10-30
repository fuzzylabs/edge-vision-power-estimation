# Torch-TensorRT

Torch-TensorRT library provides easy interface to create and run inference using TensoRT engine for PyTorch models.

## Background

[PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/) introduced features such as TorchDynamo, TorchInductor and TorchCompile.

Briefly, I add snippets from PyTorch documentation for relavent concepts above.

> `torch.compile` is a PyTorch function introduced in PyTorch 2.x that aims to solve the problem of accurate graph capturing in PyTorch and ultimately enable software engineers to run their PyTorch programs faster.

`torch.compile` leverages the following underlying technologies:

- **TorchDynamo** (`torch._dynamo` or simply Dynamo) is a Python-level Just-In-Time (JIT) compiler designed to make unmodified PyTorch programs faster. Dynamo hooks into the frame evaluation API in CPython ([PEP 523](https://peps.python.org/pep-0523/)) to dynamically modify Python bytecode right before it is executed. It rewrites Python bytecode to extract sequences of PyTorch operations into an [FX Graph](https://pytorch.org/docs/stable/fx.html) which is then compiled with a customizable backend. It creates this FX Graph through bytecode analysis and is designed to mix Python execution with compiled backends to get the best of both worlds — usability and performa

- **TorchInductor** is the default `torch.compile` deep learning compiler that generates fast code for multiple accelerators and backends. You need to use a backend compiler to make speedups through `torch.compile` possible. For NVIDIA, AMD and Intel GPUs, it leverages OpenAI Triton as the key building block.

- **AOT Autograd** captures not only the user-level code, but also backpropagation, which results in capturing the backwards pass “ahead-of-time”. This enables acceleration of both forwards and backwards pass using TorchInductor.

Refer to the guide for more information on `torch.compile`: <https://pytorch.org/docs/main/torch.compiler.html> and a video on [Deep Dive on TorchDynamo](https://www.youtube.com/watch?v=5FNHwPIyHr8&list=PL_lsbAsL_o2CQr8oh5sNWt96yWQphNEzM&index=4).

## How it works?

There are several approaches inside TorchTensorRT to get a TensorRT engine and perform inference using the same.

### TorchDynamo (Default)

There are two sub-approaches to use Dynamo as frontend.

1. `torch.compile`: It uses just-in-time (JIT) compilation, deferring compilation until the first run for greater runtime flexibility. Under the hood, it creates sub-graphs which are further partitioned into components that will run in PyTorch and ones to be further compiled to TensorRT based on support for operators. An example of using `torch.compile` approach: <https://pytorch.org/TensorRT/dynamo/torch_compile.html>.

2. `torch_tensorrt.dynamo.compile` : It performs Ahead-of-Time (AOT) compilation, meaning the model is compiled into a TensorRT engine before execution. Similar to above, it creates sub-graphs that can be partitioned to be ran in PyTorch or using TensorRT based on TensorRT operator support. An example of using `torch_tensorrt.dynamo.compile` approach: <https://pytorch.org/TensorRT/dynamo/dynamo_export.html>. This approach is what we use for our implementation.

> Note: Quantization/INT8 support is slated for a future release; currently, we support FP16 and FP32 precision layers.

### TorchScript

TorchScript is an intermediate representation of PyTorch code that allows them to be serialized and optimized for deployment. There are two sub-approaches to convert PyTorch code into Torchscript format: **tracing** and **scripting**.

The TorchScript frontend was the original default frontend for Torch-TensorRT and targets models in the TorchScript format. The graph provided will be partitioned into supported and unsupported blocks. Supported blocks will be lowered to TensorRT and unsupported blocks will remain to run with LibTorch.

> Note: TorchScript frontend (`torch_tensorrt.ts.compile`) approach is legacy feature of Torch-TensorRT library.

### FX Graph Modules

> This frontend has almost entirely been replaced by the Dynamo frontend which is a superset of the features available though the FX frontend. The original FX frontend remains in the codebase for backwards compatibility reasons.

User guide on this approach: <https://pytorch.org/TensorRT/fx/getting_started_with_fx_path.html>. I found it bit hard to digest.

## Notes and Limitations

Exploring and reading [Torch-Tensorrt documentation](https://pytorch.org/TensorRT/), I found that there are 3 approaches to converting a torch model to TensorRT.

- Dynamo Frontend
- TorchScript Frontend
- FX Frontend

It is not particularly clear which approach to prefer for a new user. For this experiment, I used `Dyanmo` approach that uses a FX Graph Module to create a TensorRT module. (Update: More clear spending more time going through and writing it down, TorchScript and FX frontend are legacy frontends and they are present for backward compatibility. Dynamo is the approach that should be used.)

Additionally, there are two approaches to create tensorrt module using `Dynamo`

- `torch.compile` [approach](https://pytorch.org/TensorRT/dynamo/torch_compile.html): This approach did not provide a way to get the underlying tensorrt engine.

- [Exported program approach](https://pytorch.org/TensorRT/dynamo/dynamo_export.html) which is what we use as our approach that provides a way to access the converted tensorrt module.

There were few notes/limitations regarding `Torch-Tensorrt` library.

- Dynamo approach does not provide `INT` precision support at the moment.

- `use_python_runtime` parameter to the compiler changes which profiler is being used.
  - If set to `True`, it uses [PythonTorchTensorRTModule](https://github.com/pytorch/TensorRT/blob/d11ff5c14cb45c975b4a9698b211ebacf1a36bb7/py/torch_tensorrt/dynamo/runtime/_PythonTorchTensorRTModule.py#L26C7-L26C32) as it's runtime. This approach _does not_ provide an option in it's [enable_profiling](https://github.com/pytorch/TensorRT/blob/d11ff5c14cb45c975b4a9698b211ebacf1a36bb7/py/torch_tensorrt/dynamo/runtime/_PythonTorchTensorRTModule.py#L417) function to save the layer-wise latency. It instead just prints the traces on the stdout. In our implementation, we have written [utility functions](./trt_utils.py) around this implementation to save layer-wise latency and tensorrt engine information to JSON files.
  - If set to `False`, it uses [TorchTensorRTModule](https://github.com/pytorch/TensorRT/blob/d11ff5c14cb45c975b4a9698b211ebacf1a36bb7/py/torch_tensorrt/dynamo/runtime/_TorchTensorRTModule.py#L53) as it's runtime. This approach _does_ provide [option](https://github.com/pytorch/TensorRT/blob/d11ff5c14cb45c975b4a9698b211ebacf1a36bb7/py/torch_tensorrt/dynamo/runtime/_TorchTensorRTModule.py#L283) to store the profiling traces in a directory.

- At present there's no clean way to profile as noted in one of the issues on the TensorRT repo : <https://github.com/pytorch/TensorRT/issues/1467>. We use a hacky approach to enable profiling as suggested in the comment on the issue.

- There are different modes of [ProfilingVerbosity](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#engine-inspector) that shows how detailed information for a particular layer is recorded. Apparently for `dynamo` approach, this is configured implicitly via `debug` parameter [here](https://github.com/pytorch/TensorRT/blob/d11ff5c14cb45c975b4a9698b211ebacf1a36bb7/py/torch_tensorrt/dynamo/conversion/_TRTInterpreter.py#L214).

- There are various parameter like `min_block_size` and `optimization` that affect the TensorRT run times.

- For example, to see if all operators are supported, we set `min_block_size` value to some higher value `100`. This shows how many graph fragmentation occurs and what is recommended value.

    Here's an example output from logs for the `resnet18` model.

    ```bash
    Graph Structure:

    Inputs: List[Tensor: (1, 3, 224, 224)@float16]
        ...
        TRT Engine #1 - Submodule name: _run_on_acc_0
        Engine Inputs: List[Tensor: (1, 3, 224, 224)@float16]
        Number of Operators in Engine: 89
        Engine Outputs: Tensor: (1, 1000)@float16
        ...
    Outputs: List[Tensor: (1, 1000)@float16]

    ------------------------- Aggregate Stats -------------------------

    Average Number of Operators per TRT Engine: 89.0
    Most Operators in a TRT Engine: 89

    ********** Recommendations **********

    - For minimal graph segmentation, select min_block_size=89 which would generate 1 TRT engine(s)
    - The current level of graph segmentation is equivalent to selecting min_block_size=89 which generates 1 TRT engine(s)
    ```

- `optimization_level` parameter decides how long to spend searching for an optimal path to minimize run time for a build. It's value ranges from 0 to 5 (higher levels imply longer build time).

### Questions

- How does setting the recommended value of `min_block_size` differ to the default? What is the perfomance difference in TensorRT inference

- Same question about `optimization_level` parameter.

- What approach should be used for integer quantization?
