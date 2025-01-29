"""Utility to dump imagenet data for calibration."""

import os

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO, settings
from ultralytics.data import YOLODataset
from ultralytics.data.utils import check_det_dataset

# View all settings
settings["datasets_dir"] = os.getcwd()
settings["weights_dir"] = os.getcwd()
settings["runs_dir"] = os.getcwd()

NUM_CALIB_IMAGES = 500


def main():
    pt_model = YOLO(model="yolov5su.pt", task="detect")

    # Download COCO val 2017 dataset
    data = check_det_dataset("cfg/coco.yaml")

    # Use only subset of data for calibration
    with open("datasets/coco/val2017.txt", "r") as fp:
        data = fp.readlines()
    data = data[:NUM_CALIB_IMAGES]
    with open("datasets/coco/val2017.txt", "w") as fp:
        fp.writelines(data)

    batch_size = 1
    dataset = YOLODataset(
        data["val"],
        data=data,
        task=pt_model.task,
        imgsz=pt_model.args["imgsz"],
        augment=False,
        batch_size=batch_size,
    )
    calib_tensor = []
    for data in tqdm(dataset):
        calib_tensor.append(data["img"].float())
    calib_tensor = np.stack(calib_tensor, axis=0)
    np.save("calib.npy", calib_tensor)


if __name__ == "__main__":
    main()
