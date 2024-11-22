# Power Consumption Modelling

This package trains layer-specific models for power consumption and runtime.

## Models

`model` module contains common code for modelling layers' power consumption and runtime, as well as pre-defined pipelines for the following layer types:

- Convolutional 2D
- Dense
- Pooling 2D

## Getting Started

Create virtual environment using `uv` and install dependencies required for the project.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

### Data Versioning

We use DagsHub and DVC to data version control

There are two operations that we can perform for a data versioning.

1. Upload the local data to be version control into DagsHub
2. Download the version control dataset from DagsHub locally

Upload dataset from `raw_data` folder to DagsHub.

```bash
python data_version.py \
    --owner DAGSHUB_USERNAME \
    --name DAGSHUB_REPONAME \
    --local-dir-path raw_data \
    --commit "Add raw data" \
    --upload
```

Download dataset from `raw_data` folder from DagsHub locally.

```bash
python data_version.py \
    --owner DAGSHUB_USERNAME \
    --name DAGSHUB_REPONAME \
    --local-dir-path raw_data \
    --remote-dir-path raw_data \
    --download
```

### Data preprocessing

Next, we will perform data preprocessing to prepare training dataset.

1. Map power to layer.

    ```bash
    python map_power_to_layers.py \
        --idling-power-log-path results/60_seconds_idling_power_log_20241103-144950.log \
        --power-log-path results/mobilenet_v2/30000_cycles_power_log_20241103-151221.log \
        --trt-layer-latency-path results/mobilenet_v2/trt_profiling/trt_layer_latency.json \
        --trt-engine-info-path results/mobilenet_v2/trt_profiling/trt_engine_info.json \
        --result-dir results
    ```

2. Prepare training data.

    ```bash
    python convert_measurements.py \
        data/mobilenet_v2 \
        results/mobilenet_v2/power_runtime_mapping_layerwise.csv \
        results/mobilenet_v2/trt_profiling/trt_engine_info.json
    ```

The final preprocessed data for the model is present in [data/model-name](data) folder.

### Training Script

#### Configuration

There are 4 configuration files under [config](./config/) folder

```bash
.
├── config.yaml        # Configuration for training pipelines
├── convolutional_features.py  # Configuration related to convolutional layer
├── dense_features.py  # Configuration related to dense layer
├── pooling_features.py # Configuration related to pooling layer
```

`config.yaml` contains data and mlflow related configuration.

Each individual `*_features.py` contains features and pipeline configuration.

#### Running the script

[run.py](./run.py) script trains power and runtime models for each layer (`convolutional`, `pooling` and `dense`).

It also logs the experiment to DagsHub MLflow server.

> [!NOTE]  
> MLFLOW UI: <https://dagshub.com/fuzzylabs/edge-vision-power-estimation.mlflow/>

```python
python run.py
```

### Training Notebook

[train.ipynb](./train.ipynb) shows how the pipelines can be trained and used.

```bash
jupyter notebook
```
