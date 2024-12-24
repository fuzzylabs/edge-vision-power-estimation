# DVC

[DVC](https://dvc.org/) offers git-like experience to track your datasets. DVC allows tracked the dataset to be uploaded to a various storage systems also known as "remotes".

We are using DagsHub (which in turn uses S3) as a remote for our DVC. [DagsHub](https://dagshub.com/) is a platform that allows data scientists and ML developers to manage and collaborate on data, models and experiments.

## Authentication

To use DVC with DagsHub, we have to setup the credentials related to DVC remote. This helps DVC know where to store and version the datasets.

The GIF below shows the required credentials to be setup locally. Run the commands at the root of the project corresponding to the `Add a DagsHub DVC remote` and `Setup credentials` sections.

![dvc_dagshub](../assets/dvc-remote.gif)

## Downloading

## Uploading

## Tagging
