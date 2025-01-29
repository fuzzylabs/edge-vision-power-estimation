import argparse

import cv2
import onnxruntime as ort
import torch
from ultralytics.utils.checks import check_requirements

from onnx_utils import OnnxYOLO

if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="yolov5su",
        help="Input your model name.",
    )
    parser.add_argument(
        "--img",
        type=str,
        default="datasets/coco/images/val2017/000000000139.jpg",
        help="Path to input image.",
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.5, help="Confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.5, help="NMS IoU threshold"
    )
    args = parser.parse_args()

    # Check the requirements and select the appropriate backend (CPU or GPU)
    check_requirements(
        "onnxruntime-gpu" if torch.cuda.is_available() else "onnxruntime"
    )

    # # Create an instance of the YOLOv8 class with the specified arguments
    # detection = OnnxYOLO(
    #     f"{args.model}.onnx", args.img, args.conf_thres, args.iou_thres
    # )
    # output_image = detection.main()
    # cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    # cv2.imshow("Output", output_image)
    # cv2.waitKey(0)

    # # Create an instance of the YOLOv8 class with the specified arguments
    # detection = OnnxYOLO(
    #     f"{args.model}.quant.onnx", args.img, args.conf_thres, args.iou_thres
    # )
    # output_image = detection.main()
    # cv2.namedWindow("Output_quant", cv2.WINDOW_NORMAL)
    # cv2.imshow("Output_quant", output_image)
    # cv2.waitKey(0)

    # Check tolerance on random input tensors for quantized and unquantized model
    x = torch.randn(size=(1, 3, 640, 640))
    print(x.shape)
    ort_sess = ort.InferenceSession("yolov5su.onnx")
    outputs = ort_sess.run(None, {"images": x.numpy()})

    ort_sess = ort.InferenceSession("yolov5su.quant.onnx")
    outputs_quant = ort_sess.run(None, {"images": x.numpy()})
    breakpoint()
