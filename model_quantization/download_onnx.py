import os

from ultralytics import YOLO, settings

# View all settings
settings["datasets_dir"] = os.getcwd()
settings["weights_dir"] = os.getcwd()
settings["runs_dir"] = os.getcwd()


def export_to_onnx(model_name: str = "yolov5su.pt"):
    pt_model = YOLO(model=model_name, task="detect")
    pt_model.export(format="onnx")


if __name__ == "__main__":
    export_to_onnx()
