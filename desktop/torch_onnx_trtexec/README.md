# PyTorch-ONNX-trtexec

In this approach, we use two step process to benchmark TensorRT model

- First convert PyTorch model to ONNX using [torch](https://pytorch.org/docs/stable/index.html) library.
- Create ONNX model to TensorRT engine and profile the engine using [trtexec](https://github.com/NVIDIA/TensorRT/tree/master/samples/trtexec) CLI.

Tested this on machine with following configuration

```txt
Python - 3.11
uv - 0.4.25
GPU - Nvidia GeForce RTX 3060 Mobile
OS - Ubuntu 22.04.5 LTS
```

An introduction documentation to `trtexec` : [Readme](./docs/trtexec.md).

## Getting Started

### Locally

Requires `trtexec` binary to be present on local machine and added to `$PATH`.

Steps to install : <https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing>

Once `trtexec` is available, it can be confirmed running `trtexec --help` command.

Install all the dependencies of the project,

```bash
uv pip install -r pyproject.toml
```

Once dependencies are install, following [benchmarking](#benchmarking) section replacing `python` with `uv run` command to run benchmarking script.

> Note: We recommend following the [Docker](#docker) approach.

### Docker (Recommended)

Build and run docker image using [Dockerfile](./Dockerfile)

```bash
docker build -t benchmark_trtexec .
docker run --gpus all -it -v $(pwd):/app benchmark_trtexec
```

> Note: Size of Docker image `nvcr.io/nvidia/pytorch:24.09-py3` is around 9 GB.

## Benchmarking

```bash
# By default, alexnet model
python benchmark.py

# For mobilenet v2 model
python benchmark.py --model mobilenet_v2

# For Resnet18 model
python benchmark.py --model resnet18
```

Running this benchmark creates following files in `models` and `results` directory.

```bash
# Saved in models dir
.engine - the built engine file.
.build.log - trtexec engine building log.
.profile.log - trtexec engine profiling log.

# Saved in results dir
.build.metadata.json - JSON of metadata parsed from the build log.
.graph.json - JSON of engine graph.
.profile.json - JSON of engine layers profiling.
.profile.metadata.json - JSON of metadata parsed from the profiling log.
.timing.json - JSON of engine profiling iteration timing.
```

## Notebook

Once benchmarking is complete, we can analyse the profiling results using `trex` library.

```bash
uv run jupyter lab
```

> Recommend using `jupyterlab` for `plotly` plots and installing [plotly](https://plotly.com/python/getting-started/#jupyterlab-support) extension for jupyterlab.

For more details on the notebook, refer [README](./notebooks/README.md) guide.
