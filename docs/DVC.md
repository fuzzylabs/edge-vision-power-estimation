# DVC

[DVC](https://dvc.org/) offers git-like experience to track your datasets. DVC allows tracked the dataset to be uploaded to a various storage systems also known as "remotes".

We are using DagsHub (which in turn uses S3) as a remote for our DVC. [DagsHub](https://dagshub.com/) is a platform that allows data scientists and ML developers to manage and collaborate on data, models and experiments.

## Authentication

To use DVC with DagsHub, we have to setup the credentials related to DVC remote. This helps DVC know where to store and version the datasets.

The GIF below shows the required credentials to be setup locally. Run the commands at the root of the project corresponding to the `Add a DagsHub DVC remote` and `Setup credentials` sections.

![dvc_dagshub](../assets/dvc-remote.gif)

## Downloading

If the DVC [authentication](#authentication) is setup correctly, downloading dataset from DagsHub is as easy as running the following command.

```bash
dvc pull -r origin
```

We can also pull a particular dataset. For example, we run the following command to pull only the `training_dataset` from DagsHub using DVC.

```bash
dvc pull -r origin training_data
```

## Uploading

We follow a particular workflow to upload and version our datasets to the DagsHub repository.

1. Create a new branch using git.
2. Track the dataset to be versioned using `dvc add` command.
3. Add a clear git commit message related to this new data version.
4. Push the data to remote using `dvc push -r origin` command.
5. Similarly, track the new `.dvc` folder using `git push origin <branch_name>` command.

**Raw Dataset**: For steps required to push raw dataset from Jetson device to DagsHub, refer to the step 5 `Upload benchmark data to DagsHub from Jetson from the current working directory` in the [jetson readme](../jetson/power_logging/README.md#-run-experiment-script).

**Training Dataset**: For steps required to push training dataset to DagsHub, refer to the [Push training dataset to DagsHub](../model_training/README.md#push-training-dataset-to-dagshub) section in the model training readme.

## Tagging

Once the dataset is uploaded, we create a PR on the DagsHub repository. This PR is then reviewed and merged into the main/develop branch.

After the dataset related changes are merged, we create a tag for that dataset using git.

Following command creates a tag for raw dataset. Make sure to add appropriate message and tag.

```bash
git tag -a raw/v1 -m "version 1 raw dataset"
git push origin raw/v1
```

Following command creates a tag for train dataset. Make sure to add appropriate message and tag.

```bash
git tag -a train/v1 -m "version 1 train dataset"
git push origin train/v1
```
