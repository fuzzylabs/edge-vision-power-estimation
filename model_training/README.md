# Model Training

Train power and runtime prediction models.

## 🔗 Quick Links

1. [Approach](#-approach)
2. [Repository Structure](#-repository-structure)
3. [Getting Started](#-getting-started)
4. [How it works?](#-how-it-works)
5. [Extras](#️-extras)

## 💡 Approach

## 🛸 Getting Started

### ⚙️ System and Hardware Requirements

- [uv](https://docs.astral.sh/uv/) : It is used as default for running this project locally.

    Create virtual environment using `uv` and install dependencies required for the project.

    ```bash
    uv venv --python 3.12
    source .venv/bin/activate
    uv sync
    ```

### 🏎💨  Run Training Script

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

2. We are all set to train power and runtime prediction models.

    ```bash
    python run.py
    ```

    That's it. We have successfully trained 6 models for 3 layer types (convolutional, pooling and dense).

---

## ❓ How it works?

## 📂 Repository Structure

## 🗣️ Extras
