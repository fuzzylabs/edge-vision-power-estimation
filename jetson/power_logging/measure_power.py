"""Script for collecting power usage during the inference cycle of a given CNN model.."""

from multiprocessing import Process, Event
from multiprocessing.synchronize import Event as EventClass
import argparse
from pathlib import Path
from datetime import datetime


def power_logging(event: EventClass, args: argparse.Namespace) -> None:
    """
    Read voltage, current and power from sys file.

    Args:
        event: An object that manages a flag for communication among processes.
        args: Arguments from CLI.
    """
    Path(args.result_dir).mkdir(exist_ok=True, parents=True)

    logs = []

    try:
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
    except KeyboardInterrupt:
        pass
    finally:
        with open(f"{args.result_dir}/power_log.log", "w") as f:
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
