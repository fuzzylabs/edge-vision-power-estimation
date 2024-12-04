# Configuration

To run model training, the following configurations are supported:

- [Training Configuration](#training-configuration)
- [Convolutional Layer Configuration](#convolutional-layer-configuration)
- [Dense Layer Configuration](#dense-layer-configuration)
- [Pooling Layer Configuration](#pooling-layer-configuration)

All configurations are stored under the [`config`](../model_training/config/) folder for the model training. The individual layer type configuration is stored under `<layer_type>_feature.py`.

## Training Configuration

The configuration file in [`config.yaml`](../model_training/config/config.yaml) consists of the following content:

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

- `data`: The data configuration provides a way to configure the location of training data on local filesystem and CNN models to be used as test dataset.

- `model`: The model configuration contains the number of cross validations to perform

- `mlflow`: The mlflow configuration provides parameters to either enable or disable MLFlow tracking. Additionally, if tracking is enabled it provides configuration for storing and tracking the experiments. By default, we use DagsHub MLflow server.

The configuration is present for our DagsHub repo. You can also modify it to run and store experiments on your DagsHub repo.

## Convolutional Layer Configuration

[`convolutional_features.py`](../model_training/config/convolutional_features.py) consists of following

- The features to be used as input (`CONV_FEATURES`)
- Special features derived using input features (`TOTAL_CONV_OPS_PER_INPUT`, `TOTAL_CONV_OPS_PER_BATCH`)
- Model Pipeline configuration such as whether to use log scaling, which scaler to use from sklearn and parameter to be passed to LassoCV model. (`CONVOLUTION_PIPELINE`)

## Dense Layer Configuration

[`dense_features.py`](../model_training/config/dense_features.py) consists of following

- The features to be used as input (`DENSE_FEATURES`)
- Model Pipeline configuration such as whether to use log scaling, which scaler to use from sklearn and parameter to be passed to LassoCV model. (`DENSE_PIPELINE`)

## Pooling Layer Configuration

[`pooling_features.py`](../model_training/config/pooling_features.py) consists of following

- The features to be used as input (`POOLING_FEATURES`)
- Special features derived using input features (`TOTAL_POOLING_INPUT_FEATURES`, `TOTAL_POOLING_NO_OPS`, `TOTAL_POOLING_OUTPUT_FEATURES`)
- Model Pipeline configuration such as whether to use log scaling, which scaler to use from sklearn and parameter to be passed to LassoCV model. (`POOLING_PIPELINE`)
