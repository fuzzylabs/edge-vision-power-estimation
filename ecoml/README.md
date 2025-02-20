# EcoML

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

EcoML aims to provide accurate power consumption estimates for edge devices, helping developers optimize their applications for energy efficiency.

## Installation

To install the necessary dependencies, run the following command:

```bash
uv venv
source .venv/bin/activate
uv sync
```

> [!NOTE]
> Coming Soon: We will also publish the package on PyPI for ease of use.

## Usage

To use EcoML for energy estimation of PyTorch models, follow these steps:

1. Using PyTorch model summary

    ```bash
    ecoml predict --model sample_data/resnet18.json
    ```

    [Sample data](./sample_data/) folder contains model summary for 3 PyTorch models - Resnet18, Mobilenetv2 and VGG16.

    To use a custom model for inference, you have to generate a model summary for the PyTorch model.

    > [!TIP]
    > [save_model_summary.py](https://github.com/fuzzylabs/ecomlops/blob/develop/jetson/power_logging/save_model_summary.py) script can be used to create a model summary for a custom PyTorch model.

    `--verbose` flag can be passed to above command to get a detailed output.

2. Using custom PyTorch model in your workflow

    ```bash
    from ecoml.model_summary import get_summary

    summary = get_summary(your_pt_model, model_input_shape, summary_file_path='summary/my_model.json')
    ```

    Here `your_pt_model` is a instance `nn.Module`, the trained PyTorch model.

    Next, you can use the `predict` command to get the energy prediction using the path where model summary is saved.

   ```bash
    ecoml predict --model summary/my_model.json
    ```
