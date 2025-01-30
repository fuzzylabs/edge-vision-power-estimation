import os

# import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
# import torch
from ultralytics import YOLO, settings
from ultralytics.data import YOLODataset, build_dataloader
from ultralytics.data.utils import check_det_dataset

NUM_CALIB_IMAGES = 500

# View all settings
settings["datasets_dir"] = os.getcwd()
settings["weights_dir"] = os.getcwd()
settings["runs_dir"] = os.getcwd()


def quantized_pt_model():
    # Setup the model
    pt_model = YOLO(model="yolov5su.pt", task="detect")

    # Select quantization config
    config = mtq.INT8_SMOOTHQUANT_CFG

    # Quantization need calibration data. Setup calibration data loader
    # Download COCO val 2017 dataset
    data = check_det_dataset("cfg/coco.yaml")

    # Use only subset of data for calibration
    with open("datasets/coco/val2017.txt", "r") as fp:
        val_data = fp.readlines()
    val_data = val_data[:NUM_CALIB_IMAGES]
    with open("datasets/coco/val2017.txt", "w") as fp:
        fp.writelines(val_data)

    batch_size = 1
    dataset = YOLODataset(
        data["val"],
        data=data,
        task=pt_model.task,
        imgsz=pt_model.args["imgsz"],
        augment=False,
        batch_size=batch_size,
    )

    data_loader = build_dataloader(dataset, batch=batch_size, workers=0)

    # Define forward_loop.
    def forward_loop(model):
        for batch in data_loader:
            model(batch["img"].float() / 255.0)

    # Quantize the model and perform calibration (PTQ)
    qt_model = mtq.quantize(pt_model.model, config, forward_loop)
    return qt_model


# The quantized pytorch model cannot be pickled, traced or scripted to be saved.
# We can save the model weights using qt_model.state_dict() but we don't have the class to assign the weights to.
# It has to be exported to ONNX to be usable.

# All different approaches tested for saving quantized pytorch model

# torch.save(qt_model.state_dict(), "yolov5su.quant.pt")

# mo.save(qt_model, "yolov5su.quant.pt")

# model_scripted = torch.jit.script(qt_model)
# model_scripted.save("yolov5su.quant.pth")

# model_trace = torch.jit.script(qt_model, torch.randn(1, 3, 640, 640) / 255.0)
# torch.jit.save(model_trace, "yolov5su.quant.pt")


if __name__ == "__main__":
    quantized_pt_model()
