# Configuration

To run model training, following configuration are supported

- [Training Configuration](#training-configuration)
- [Convolutional Layer Configuration](#convolutional-layer-configuration)
- [Dense Layer Configuration](#dense-layer-configuration)
- [Pooling Layer Configuration](#pooling-layer-configuration)

All configuration are stored under [`config`](../model_training/config/) folder for the model training. The individual layer type configuration is stored under `<layer_type>_feature.py`.

## Training Configuration

The configuration file in [`config.yaml`](../model_training/config/config.yaml) consists of following content

```yaml
data:
  data_dir: "training_data"
  test_models: ["lenet", "resnet18", "vgg16"]

model:
  cross_validation: 10

mlflow:
  enable_tracking: False
  mlflow_experiment_name: "second-data-version-exp"
  dagshub_repo_owner: "fuzzylabs"
  dagshub_repo_name: "edge-vision-power-estimation"

```

- `data`: The data configuration provides a way to configure location of training data on local filesystem and CNN models to be used as test dataset.

- `model`: The model configuration contains number of cross validation to perform

- `mlflow`: The mlflow configuration provides parameter to either enable or disable mlflow tracking. Additionally, if tracking is enabled it provides configuration for storing and tracking the experiments. By default, we use DagsHub MLflow server.

The configuration is present for our DagsHub repo. You can also modify it to run and store experiments on your DagsHub repo.

## Convolutional Layer Configuration

[`convolutional_features.py`](../model_training/config/convolutional_features.py) consists of following

- The features to be used as input
- Special features derived using input features
- Model Pipeline configuration such as whether to use log scaling, which scaler to use from sklearn and parameter to be passed to LassoCV model.

## Dense Layer Configuration

[`dense_features.py`](../model_training/config/dense_features.py) consists of following

- The features to be used as input
- Special features derived using input features
- Model Pipeline configuration such as whether to use log scaling, which scaler to use from sklearn and parameter to be passed to LassoCV model.

## Pooling Layer Configuration

[`pooling_features.py`](../model_training/config/pooling_features.py) consists of following

- The features to be used as input
- Special features derived using input features
- Model Pipeline configuration such as whether to use log scaling, which scaler to use from sklearn and parameter to be passed to LassoCV model.
