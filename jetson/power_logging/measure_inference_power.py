"""Script for collecting power usage during the inference cycle of a given CNN model.."""

from multiprocessing import Process, Event
from multiprocessing.synchronize import Event as EventClass
import argparse
from pathlib import Path
from datetime import datetime
from model.benchmark import benchmark
import multiprocessing

multiprocessing.set_start_method('spawn', force=True)


def power_logging(event: EventClass, args: argparse.Namespace) -> None:
    """
    Read voltage, current and power from sys file.

    Args:
        event: An object that manages a flag for communication among processes.
        args: Arguments from CLI.
    """
    Path(args.result_dir).mkdir(exist_ok=True, parents=True)

    logs = []

    while not event.is_set():
        with open("/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/in1_input", "r") as voltage:
            mV = float(voltage.read())
        with open("/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/curr1_input", "r") as current:
            mC = float(current.read())

        current_time = datetime.now().strftime("%Y%m%d-%H:%M:%S.%f")  # Time with seconds and microseconds
        logs.append(f"{current_time},{mV},{mC}\n")  # Log the time, voltage and current.

    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f"{args.result_dir}/power_log_{current_dt}.log", "w") as f:
        f.writelines(logs)


def inference(
    event: EventClass,
    args: argparse.Namespace,
) -> None:
    """
    Call the benchmark function to start inferencing cycles.

    Save layer wise latency with time stamp.

    Args:
        event: An object that manages a flag for communication among processes.
        args: _Arguments from CLI.
    """
    benchmark(args)
    event.set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Power Logging for CNN Inference Cycle",
        description="Collect power usage data during inference cycles for ImageNet pretrained CNN models."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet_v2",
        help="Specify name of pretrained CNN model from PyTorch Hub."
        "For more information on PyTorch Hub visit: "
        "https://pytorch.org/hub/research-models",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model weights and activations.\n\n"
        '* "float16" is the same as "half".\n'
        '* "bfloat16" for a balance between precision and range.\n'
        '* "float32" for FP32 precision.',
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs="+",
        default=[1, 3, 224, 224],
        help="Input shape BCHW",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of iterations to perform warmup before benchmarking",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=30000,
        help="Number of inference cycle to run"
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=5,
        help="Builder optimization 0-5, higher levels imply longer build time, "
        "searching for more optimization options.",
    )
    parser.add_argument(
        "--min-block-size",
        type=int,
        default=5,
        help="Minimum number of operators per TRT-Engine Block",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="results",
        help="The directory to save the log result."
    )
    args = parser.parse_args()

    event = Event()
    power_logging_process = Process(target=power_logging, args=(event, args))
    power_logging_process.start()

    inference_process = Process(target=inference, args=(event, args))
    inference_process.start()

    power_logging_process.join()
    inference_process.join()
