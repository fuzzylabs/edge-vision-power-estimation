# EcoML

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

EcoML aims to provide accurate power consumption estimates for various edge devices, helping developers optimize their applications for energy efficiency. By leveraging machine learning, EcoML can predict power usage based on different workloads and device configurations.

## Installation

To install the necessary dependencies, run the following command:

```bash
python3 -m venv venv
source venv/bin/activate
pip install ecoml
```

## Usage

To use EcoML for energy estimation of PyTorch models, follow these steps:

```bash
 ecoml predict --model sample_data/resnet18.json
```

[Sample data](./sample_data/) folder contains model summary for 3 PyTorch models - Resnet18, Mobilenetv2 and VGG16.

To use a custom model for inference, you have to generate a model summary for the PyTorch model.

> [!TIP]
> [save_model_summary.py](https://github.com/fuzzylabs/ecomlops/blob/develop/jetson/power_logging/save_model_summary.py) script can be used to create a model summary for a custom PyTorch model.

`--verbose` flag can be passed to above command to get a detailed output.
