"""Script for collecting power usage during the inference cycle of a given CNN model.."""

from multiprocessing import Process, Event
from multiprocessing.synchronize import Event as EventClass
from time import perf_counter
import argparse
from pathlib import Path


def power_logging(event: EventClass, args: argparse.Namespace) -> None:
    """
    Read voltage, current and power from sys file.

    Args:
        event: An object that manages a flag for communication among processes.
        args: Arguments from CLI.
    """
    Path(args.result_dir).mkdir(exist_ok=True, parents=True)

    logs = []
    start_time = perf_counter()

    while not event.is_set():
        with open("/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/in1_input", "r") as vdd_in: # vdd_in is global power consumption of the board
            mW = float(vdd_in.read())

        timestamp = perf_counter() - start_time
        logs.append(f"{timestamp},{mW}\n")

        # For now stop after 5 seconds
        if perf_counter() - start_time > 5:
            event.set()
        # The above will be moved to the inference function later
        # where an event will be set after inference has finished.

    with open(f"{args.result_dir}/{start_time}.log", "w") as f:
        f.writelines(logs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Power Logging for CNN Inference Cycle",
        description="Collect power usage data during inference cycles for ImageNet pretrained CNN models."
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
    power_logging_process.join()
