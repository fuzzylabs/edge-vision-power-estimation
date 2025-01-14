# Model Training

Train power and runtime prediction models.

## 🔗 Quick Links

1. [Getting Started](#-getting-started)
2. [Approach](#-approach)
3. [Repository Structure](#-repository-structure)
4. [Documentation](#-documentation)

## 🛸 Getting Started

### ⚙️ Requirements

[uv](https://docs.astral.sh/uv/) : It is used as default for running this project locally.

Create virtual environment using `uv` and install dependencies required for the project.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

You can run the preprocessing and training scripts on your laptop/desktop locally.

---

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

Trained models are downloaded in thee `trained_models` folders. This is how the tree for the `trained_models` folder looks like with all the models downloaded

```bash
trained_models
├── convolutional
│   ├── power
│   └── runtime
├── dense
│   ├── power
│   └── runtime
└── pooling
    ├── power
    └── runtime
```

For each of the layer types (convolutional, dense and pooling), a power and runtime model is downloaded from the MLFlow Registry to your local filesystem.

---

### 🏎💨 Run Training Script

### Raw data is collected on Jetson

If you have uploaded the raw dataset from the benchmarking experiment on the Jetson device, the next step is to get a training dataset.

If not you can pull the raw data from DagsHub using following command,

```bash
dvc pull -r origin
```

This will create a `raw_data` folder under [`jetson/power_logging`](../jetson/power_logging/) folder containing data from our benchmarking experiment.

To process this raw dataset into training data ingestible by a model, run the [`create_dataset.sh`](./create_dataset.sh) script.

```bash
./create_dataset.sh
```

To know more about the contents in this script, refer to the [Data Preprocessing](../docs/ExperimentScripts.md#data-preprocessing-script) script section.

#### Push training dataset to DagsHub

To push the training data to DagsHub using DVC, follow the steps outlined below

1. Add DVC credentials locally as shown in the video below. Run the commands at the root of the project corresponding to the `Add a DagsHub DVC remote` and `Setup credentials` sections.

    ```bash
    $ pwd
    /home/username/edge-vision-power-estimation
    ```

    <a href="DVC Remote"><img src="../assets/dvc-remote.gif" align="center" height="500" width="500" ></a>

2. Upload training data to DagsHub from the root directory of the project.

    We create a new branch `train_data_v1`. Please make sure to add a new branch for clarity.

    ```bash
    git checkout -b train_data_v1
    ```

    Track `training_data` folder using `dvc add` command

    ```bash
    dvc add model_training/training_data
    ```

    Next, run the following commands to track changes in Git. For example, we add a commit message `Add training data version 1`. Please make sure to add a good commit message for clarity.

    ```bash
    git add model_training/training_data.dvc
    git commit -m "Add training data version 1"
    ```

    Push both the data and new git branch to the remote

    ```bash
    dvc push -r origin
    git push origin train_data_v1
    ```

After the PR related to train dataset is merged, a tag for that specific version of train dataset should be created. To know more about tagging, refer to [DVC tagging](../docs/DVC.md#tagging) documentation.

---

### Download Training Data and Run Training

**Download Training Data**: DagsHub already contains the training dataset that we can use directly. To download the latest training dataset run the following command

```bash
dvc pull training_data -r origin
```

This will download the latest training data from the FuzzyLabs [DagsHub repository](https://dagshub.com/fuzzylabs/edge-vision-power-estimation) to the `training_data` folder on your local filesystem.

Alternatively, you can also choose to download any of the older training data using git tags. For example, following command will pull training data corresponding to `train/v1` git tag.

```bash
git checkout train/v1 -- training_data.dvc
dvc checkout training_data.dvc
```

> [!NOTE]
> This step is recommended if you want to get started with training the models using data already present on DagsHub repository. </br>
> If you have a new raw dataset, follow the steps outlined in raw data collected on Jetson [section](#raw-data-is-collected-on-jetson) to create a training dataset.

**Run Training Script**: We are all set to train power and runtime prediction models.

```bash
python run.py
```

🎉 That's it. We have successfully trained 6 models for 3 layer types (convolutional, pooling and dense).

To learn more about various configuration offered as part of training script, refer [configuration](../docs/TrainingConfiguration.md) document.

## 💡 Approach

We use the raw dataset from Jetson to create a preprocessed and training dataset. The train dataset contains power and runtime measurements for 3 layer types, convolutional, pooling and dense for the CNN models.

The raw dataset that we have collected from the Jetson lives in DagsHub - running the [`create_dataset.sh`](create_dataset.sh) script orchestrates the following data pipeline:

1. Builds the pre-processed dataset by mapping power readings to individual layers in the CNN ([`map_power_to_layers.py`](map_power_to_layers.py)).
2. Reformats the pre-processed dataset into a sklearn compatible training dataset ([`convert_measurements.py`](convert_measurements.py))

![data_pipeline](assets/data_pipeline.png)

We use [LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html) model from sklearn to train our prediction models. The sklearn training pipeline contains an input feature preprocessing step for creating the polynomial degree of input features, applying sklearn preprocessing scalers and special terms to input features.

The [`run.py`](run.py) script orchestrates the following training pipeline:

1. Uses the training dataset found on the local system for training.
2. Initiates the training of 3 power consumption and 3 runtime prediction models.
3. Logs metrics and artifacts to MLFlow's experiment tracker.

![training_process](assets/training_process.png)

## 📂 Repository Structure

```bash
.
├── assets
├── config                    # Configuration required for training prediction models
├── convert_measurements.py   # Script to convert preprocessed data to training data
├── create_dataset.sh         # Script to convert raw data to train data
├── data_preparation          # Utility functions for parsing preprocessed data
├── dataset_builder           # Dataset Builder
├── map_power_to_layers.py    # Script to convert raw data to preprocessed data
├── model_builder             # Model Builder
├── notebooks                 # Notebooks containing data exploration and hyperparameter tuning
├── trainer                   # Trainer
├── pyproject.toml
├── README.md
├── run.py
└── uv.lock
```

- **[run.py](./run.py)**: Entrypoint for training prediction models.

- **[notebooks](./notebooks/)**: Notebooks folder contains jupyter notebooks for exploring data and performing hyperparameter tuning using `optuna` library.

## 📚 Documentation

Here are few links to the relevant documentation for further readings.

- [Training Configuration](../docs/TrainingConfiguration.md)
- [Preprocessed dataset format](../docs/DatasetFormats.md#preprocessing-dataset-format)
- [Training dataset format](../docs/DatasetFormats.md#training-dataset-format)
- [Data preprocessing script](../docs/ExperimentScripts.md#data-preprocessing-script)
- [HyperParameter Tuning](../docs/HyperparameterTuning.md)
- [Mapping power to runtime per-layer](../docs/DeepDive.md#mapping-power-to-layer-runtimes)
