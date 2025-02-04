"""Script for collecting power usage during the inference cycle of a given CNN model.."""

import argparse
import multiprocessing
from datetime import datetime
from multiprocessing import Event, Process
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path

from model.benchmark import benchmark

multiprocessing.set_start_method("spawn", force=True)


def power_logging(event: EventClass, args: argparse.Namespace) -> None:
    """
    Read voltage, current and power from sys file.

    Args:
        event: An object that manages a flag for communication among processes.
        args: Arguments from CLI.
    """
    model_dir = f"{args.result_dir}/{args.model}"
    Path(model_dir).mkdir(exist_ok=True, parents=True)

    logs = []

    while not event.is_set():
        with open(
            "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/in1_input", "r"
        ) as voltage:
            mV = float(voltage.read())
        with open(
            "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/curr1_input", "r"
        ) as current:
            mC = float(current.read())

        # Time with seconds and microseconds
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S.%f")
        # Log the time, voltage and current.
        logs.append(f"{timestamp},{mV},{mC}\n")

    with open(f"{model_dir}/{args.model}_power_log.log", "w") as f:
        f.writelines(logs)


def inference(event: EventClass, args: argparse.Namespace) -> None:
    """
    Call the benchmark function to start inferencing cycles.

    Save layer wise latency with time stamp.

    Args:
        event: An object that manages a flag for communication among processes.
        args: Arguments from CLI.
    """
    benchmark(args)
    event.set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Power Logging for CNN Inference Cycle",
        description="Collect power usage data during inference cycles for ImageNet pretrained CNN models.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov5n",
        help="Specify name of pretrained CNN model from ultralytics.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="coco8.yaml",
        help="Specify name of dataset from ultralytics.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="results",
        help="The directory to save the log result.",
    )
    parser.add_argument(
        "--disable-power-measurement",
        action="store_true",
        help="Disable power measurement during benchmark execution.",
    )
    parser.add_argument(
        "--use-zkp",
        action="store_true",
        help="Apply Zero-Keep Pruning",
    )
    args = parser.parse_args()

    event = Event()
    power_logging_process = Process(target=power_logging, args=(event, args))

    if not args.disable_power_measurement:
        power_logging_process.start()

    inference_process = Process(target=inference, args=(event, args))
    inference_process.start()

    if not args.disable_power_measurement:
        power_logging_process.join()

    inference_process.join()
