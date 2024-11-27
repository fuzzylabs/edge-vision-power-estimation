# Neural Power

Paper: <https://arxiv.org/abs/1710.05420>

## 🔗 Quick Links

1. [Overview](#overview)
2. [Methodology](#methodology)
3. [Results](#results)
4. [Extra](#extra)

NeuralPower paper starts with the following 3 questions as main motivation as part of their research,

1. How much energy is consumed for an inference made by a convolutional neural network (CNN)?
2. Is it possible to predict this energy consumption before a model is even trained?
3. If yes, how should machine learners select an energy-efficient CNN for deployment?

These are the 3 questions we are also interested in researching particularly for the edge devices.

## Overview

NeuralPower trains a polynomial regression model that predicts power consumption of CNNs running on GPU with an average accuracy of 88.34% and another regression model to predict runtime of CNNs that yields an improvement of 68.5% compared to previous state-of-the-art model.

## Methodology

Before looking into how to train a regression model that predicts power consumption and runtime for a CNN model, we look at how data is collected to enable the training process.

### Data Collection

The authors use a collection of Caffe - a deep learning framework - CNN models for their experiment.

> [!NOTE]
> We use [PyTorch Hub](https://pytorch.org/hub/) that contains collection of CNN models for our project.

For each model, power consumed and runtime taken for each layer in the model is recorded.

To measure runtime of each layers, they use Paleo framework, based on research by [Qi et al](https://openreview.net/pdf?id=SyVVJ85lg).

> [!NOTE]
> We create a custom TensorRT profiler using [IProfiler](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Profiler.html#tensorrt.IProfiler) TensorRT Python API to measure the runtime for each layer in the TensorRT model on the Jetson device.

Since the experiment is performed on Nvidia GeForce Titan X GPU, the authors use `nvidia-smi` to collect power measurement per 1 ms.

> [!NOTE]
> The process for power consumption measurement on the Jetson device is detailed in [Power Consumption and Benchmarking documentation](../jetson/power_logging/docs/Power_consumption.md) document.

The dataset consists of layer-level runtime and power consumption for each layer of all the models. The authors focuses on data related to 3 types of layers, particularly, convolution layer, pooling layer and dense or fully-connected layer. A power model and runtime model is trained for these 3 layers, a total of 6 models being trained.

### Training Model

Once dataset is collected for all the models of interest, a polynomial-based regression is trained. The authors outline 3 reasons for this choice:

(i) First, in terms of model accuracy, polynomial models provide more flexibility and low prediction error when modeling both power and runtime.

(ii) The second reason is the interpretability: runtime and power have clear physical correlation with the layer’s configuration parameters

(iii) The third reason is the available amount of sampling data points. Polynomial models allow for adjustable model complexity by tuning the degree of the polynomial, ranging from linear model to polynomials of high degree, whereas a formulation with larger model capacity may be prone to overfitting.

#### Layer-level Power Modelling

A polynomial regression learns the coefficients using the input data. For power modelling, various features for a layer are combined to create a n-th degree polynomial term. For e.g. if input features for dense layers  include the batch size, the input tensor size, and the output tensor size. A 2-degree polynomial will consider pair-wise combination of each feature as it's training data.

For convolution layer, a features vector consists of the batch size, the input tensor size, the kernel size, the stride size, the padding size, and the output tensor size.

For pooling layer, features include the input tensor size, the stride size, the kernel size, and the output
tensor size.

A power model is trained for each of the layer type using Lasso and cross-validation to find a polynomial degree that maximises the accuracy using the selected features.

#### Layer-level Runtime Modelling

In addition to the features included in the power modelling, runtime modelling adds special features to the regression model. The special features include the total number of memory accesses and the total number of
floating point operations for each layer.

Same methodology of using  Lasso and cross-validation is used to find optimal polynomial degree.

### Results

The figure below shows the performance of NeuralPower runtime model.

![runtime](../assets/runtime_neuralpower.png)

The figure below shows the performance of NeuralPower power model.

![power](../assets/power_neuralpower.png)

RMSPE in the figure refers to Root-Mean-Square-Percentage-Error.

## Extra

The primary author of the paper has also tested NeuralPower approach on Nvidia Jetson TX1 edge device. This research is published in [Power/Performance Modeling and Optimization: Using and Characterizing Machine Learning Applications](https://kilthub.cmu.edu/articles/Power_Performance_Modeling_and_Optimization_Using_and_Characterizing_Machine_Learning_Applications/7212224), Section 6.3.

![Results](../assets/jetson_tx1_neuralpower.png)
