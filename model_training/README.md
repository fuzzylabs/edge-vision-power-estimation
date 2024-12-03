# Model Training

Train power and runtime prediction models.

## 🔗 Quick Links

1. [Approach](#-approach)
2. [Getting Started](#-getting-started)
3. [How it works?](#-how-it-works)
4. [Repository Structure](#-repository-structure)
5. [Extras](#️-extras)

## 💡 Approach

We use the raw dataset from Jetson to create a preprocessed and training dataset. The train dataset contains power and runtime measurement for 3 layers, convolutional, pooling and dense for the CNN models.

We use [LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html) model from sklearn to train our prediction models. The sklearn pipeline contains input feature preprocessing step such as creating polynomial degree of input features, applying sklearn preprocessing scalers, adding special terms to input features.

## 🛸 Getting Started

### ⚙️ System and Hardware Requirements

- [uv](https://docs.astral.sh/uv/) : It is used as default for running this project locally.

    Create virtual environment using `uv` and install dependencies required for the project.

    ```bash
    uv venv --python 3.12
    source .venv/bin/activate
    uv sync
    ```

### 🔋 Inference

Inference script requires path to TensorRT engine information file and a path to save prediction as a CSV file.

Inference script downloads the trained model from MLFlow registry for the inference. There are two sample data for `resnet18` and `vgg16` Tensorrt engine files under [sample_data](./sample_data/) folder.

Following command runs inference for `resnet18` model,

```python
python inference.py \
    --trt-engine-path sample_data/resnet18_trt_engine_info.json \
    --result-csv-path results/resnet18_predictions.csv
```

A prediction CSV is created in the `results` folder. The power prediction is saved under the column `power_prediction` and runtime predictions are saved under the column `runtime_prediction`.

Trained models are downloaded in `trained_models` folders. This is how the tree for `trained_models` folder looks like with all the models downloaded

```bash
.
├── convolutional
│   ├── power
│   └── runtime
├── dense
│   ├── power
│   └── runtime
└── pooling
    ├── power
    └── runtime
```

For each of the layer type, we download a power and runtime model.

### 🏎💨 Run Training Script

0. (Optional) If you have uploaded the raw dataset of benchmarking experiment from Jetson device, the next step is to get a training dataset.

    To get training dataset from raw dataset, there's a script [create_dataset.sh](./create_dataset.sh) that simplifies this process.

    ```bash
    ./create_dataset.sh
    ```

    Running this script creates `preprocessed_data` and `training_data` using `raw_data`. The `raw_data` is downloaded from DagsHub. You can also pass `--push-to-dagshub` flag to the above script, this will push both the `preprocessed_data` and `training_data` to DagsHub repository.

> [!NOTE]
> Once you have the `training_data`, you can run the step 2 and skip step 1.

1. DagsHub already contains the training dataset that we can use directly. To download the latest training dataset run the following command

    ```bash
    python data_version.py \
    --owner fuzzylabs \
    --name edge-vision-power-estimation \
    --local-dir-path training_data \
    --remote-dir-path training_data \
    --branch main \
    --download
    ```

    This will download data from our [DagsHub repository](https://dagshub.com/fuzzylabs/edge-vision-power-estimation) and to a `training_data` folder.

> [!NOTE]
> This step is recommended if you want to get started with training the models using data already present on DagsHub repository. </br>
> If you have a new raw dataset, follow step 0 to create a training dataset.

2. We are all set to train power and runtime prediction models.

    ```bash
    python run.py
    ```

    That's it. We have successfully trained 6 models for 3 layer types (convolutional, pooling and dense).

## ❓ How it works?

### Mapping power to layer runtimes

This is the most tricky part of the pipeline. We are given raw dataset with time-stamped power log as a separate file and time-stamped layer runtime as a separate file. We have to find the power value that is closest to the time when that particular layer completed.

> [!NOTE]  
> This logic is implemented inside the `compute_layer_metrics_by_cycle` function of the [data_preprocess.py](./data_preparation/data_preprocess.py) python script.

## 📂 Repository Structure

```bash
.
├── assets
├── config                    # Configuration required for training prediction models
├── convert_measurements.py   # Script to convert preprocessed data to training data
├── create_dataset.sh         # Script to convert raw data to train data and upload data to DagsHub 
├── data_preparation          # Utility functions for parsing preprocessed data
├── dataset                   # Dataset Builder
├── data_version.py           # DagsHub client to upload and download data from/to DagsHub
├── map_power_to_layers.py    # Script to convert raw data to preprocessed data
├── model                     # Model Builder
├── notebooks                 # Notebooks containing data exploration and hyperparameter tuning
├── pipeline                  # Trainer
├── pyproject.toml
├── README.md
├── run.py
└── uv.lock
```

- **[run.py](./run.py)**: Entrypoint for training prediction models.

- **[notebooks](./notebooks/)**: Notebooks folder contains jupyter notebooks for exploring data and performing hyperparameter tuning on all 3 layer types.

## 🗣️ Extras

### Preprocessing and Training Scripts

[create_dataset.sh](./create_dataset.sh) script provides a helpful utility to get preprocessed and training dataset from raw dataset. The script performs the following operations

1. `Lines 5:14`: Setup the configuration required to pull raw dataset from DagsHub repository.
2. `Lines 28:36`: Pull the raw dataset from DagsHub and store it in `raw_data` folder locally.
3. `Lines 38:45`: Create preprocessed dataset from raw dataset using the `map_power_to_layers.py` script.
4. `Lines 47:56`: Optionally upload the preprocessed data to DagsHub if `--push-to-dagshub` flag is passed while running this script.
5. `Lines 58:65`: Create training dataset from preprocessed dataset using the `convert_measurements.py` script.
6. `Lines 67:76`: Optionally upload the training data to DagsHub if `--push-to-dagshub` flag is passed while running this script.

If `--push-to-dagshub` flag is passed to the script, both the preprocessed data and training data get uploaded to the DagsHub repo.

```bash
./create_dataset.sh --push-to-dagshub
```

If don't want to push any data to DagsHub, we run the script without any flags.

```bash
./create_dataset.sh
```

### Hyperparameter Tuning

[Optuna](https://optuna.readthedocs.io/en/stable/) library is used to find optimal hyperparameter for power and runtime models for 3 layer types.

The optimal parameter found in the notebook are used to update the `lasso_params` key for each of 3 layer configuration `*_features.py` inside [config](./config/) folder.

### Preprocessing Dataset Format

A `preprocessed_data` folder is created after running the `map_power_to_layers.py` script. It takes input the raw data folder, maps power and runtime values for each layer and creates a CSV.

Each model in `preprocessed_data` folder contains 2 files: `power_runtime_mapping_layerwise.csv` and `trt_engine_info.json`.

`trt_engine_info.json`: This is same JSON file that is created in the raw data. It contains detailed information such as padding, strides, dilation for a convolutional layer.

`power_runtime_mapping_layerwise.csv`: This CSV file contains per-layer data for each iteration of inference cycle. It includes information about current inference iteration, layer name, layer type, power consumed by the layer, runtime latency for the layer. An example entry of CSV for Alexnet model is shown below.

```csv
cycle,layer_name,layer_type,layer_power_including_idle_power_micro_watt,layer_power_excluding_idle_power_micro_watt,layer_run_time
1,Reformatting CopyNode for Input Tensor 0 to [CONVOLUTION]-[aten_ops.convolution.default]-[/feat_conv1/convolution] + [RELU]-[aten_ops.relu.default]-[/feat/relu],Reformat,5437392.769097799,1202832.6137757916,0.10390400141477585
```

### Training Dataset Format

A `training_data` folder is created after running `convert_measurements.py` script. It takes input the preprocessed data folder and splits the data into 3 CSV according to layer types of interest.

Each model in `training_data` folder contains 3 CSV files: `dense.csv`, `convolutional.csv` and `pooling.csv`.

Each CSV file stores relevant information such as batch_size, input_size, output_size, input_shape, output_shape, strides, padding, dilation, depending on the layer type.

An example of columns in Alexnet dense CSV is shown below

```csv
batch_size,input_size,output_size,power,runtime,layer_name
```

An example of columns in Alexnet pooling CSV is shown below

```csv
batch_size,input_size_0,input_size_1,input_size_2,output_size_0,output_size_1,output_size_2,kernel_0,kernel_1,stride_0,stride_1,power,runtime,layer_name
```

An example of columns in Alexnet convolutional CSV is shown below

```csv
batch_size,input_size_0,input_size_1,input_size_2,output_size_0,output_size_1,output_size_2,kernel_0,kernel_1,padding_0,padding_1,stride_0,stride_1,power,runtime,layer_name
```

Pooling and Convoluion store contains almost the same features except for padding in pooling layers. Dense layers store only input_size, output_size and batch_size information. All the layers in CSV contain values for power and runtime which is used for training the prediction models and evaluating the trained models.
