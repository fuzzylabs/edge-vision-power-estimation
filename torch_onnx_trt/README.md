# PyTorch-ONNX-TensorRT

We take a two-step approach for this experiment,

1. Convert PyTorch model to ONNX format using `torch.onnx` function
2. Create a TensorRT inference using the ONNX model by creating a wrapper around TensorRT Python API.

A major difference between [Torch-TensorRT](../torch_trt/README.md) experiment and current approach is we have full control over creating TensorRT engine and inference using engine.

Tested this on machine with following configuration

```txt
Python - 3.11
uv - 0.4.25
GPU - Nvidia GeForce RTX 3060 Mobile
OS - Ubuntu 22.04.5 LTS
```

## Getting Started

There are two approaches to run this project.

### Locally

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r pyproject.toml
```

### Docker

> Note: Size of Docker image `nvcr.io/nvidia/pytorch:24.09-py3` is around 9 GB.

```bash
docker run --gpus all -it --rm -v $(pwd)/:/workspace/  nvcr.io/nvidia/pytorch:24.09-py3
```

Inside docker container, we install `onnxruntime-gpu` package

```bash
pip install onnxruntime-gpu==1.19.2
```

> Note: Replace `uv` in following commands with `python` if running the script inside docker container.

### Run the benchmark script

For example let us run the benchmark script using `Mobilenetv2` model from [pytorch hub](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)

```bash
uv run benchmark.py --model mobilenet_v2 --backend onnx --save-result
```
