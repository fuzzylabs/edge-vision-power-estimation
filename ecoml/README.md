# EcoML

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

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

[Sample data](./sample_data/) contains model summary for 3 models - Resnet18, Mobilenetv2 and VGG16.

To use a custom model for inference, you have to generate a model summary for the PyTorch model.

`--verbose` flag can be passed to above command to get a detailed output.

## Contributing

We welcome contributions to EcoML! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.
