import argparse

import cv2

from utils import InferYOLO

if __name__ == "__main__":
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
    parser.add_argument(
        "--with-quant", action="store_true", help="Run inference quantization model"
    )
    args = parser.parse_args()

    print("Using original PyTorch model for inference")
    detection = InferYOLO(f"{args.model}.pt", args.img, args.conf_thres, args.iou_thres)
    detection.infer_pt()

    if args.with_quant:
        print("Using quantized PyTorch model for inference")
        detection = InferYOLO(
            f"{args.model}.quant.pt", args.img, args.conf_thres, args.iou_thres
        )
        output_image = detection.infer_pt()
        cv2.namedWindow("Output_quant", cv2.WINDOW_NORMAL)
        cv2.imshow("Output_quant", output_image)
        cv2.waitKey(0)
