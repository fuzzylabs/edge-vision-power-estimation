import argparse
import os

from ultralytics import YOLO, settings

# View all settings
settings["datasets_dir"] = os.getcwd()
settings["weights_dir"] = os.getcwd()
settings["runs_dir"] = os.getcwd()


def export_to_trt_engine(
    model_name: str = "yolov5su.pt", data_path: str = "cfg/coco.yaml"
):
    pt_model = YOLO(model=model_name, task="detect")
    pt_model.export(
        format="engine",
        data=data_path,
        int8=True,
        dynamic=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert pretrained YOLO models to TensorRT engine.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov5n.pt",
        help="Specify name of pretrained CNN model from ultralytics.",
    )
    args = parser.parse_args()
    export_to_trt_engine(args.model)
