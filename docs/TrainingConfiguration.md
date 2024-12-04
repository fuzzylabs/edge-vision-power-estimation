# Configuration

To run model training, following configuration are supported

- [Training Configuration]

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
