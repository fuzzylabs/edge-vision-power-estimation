from multiprocessing import Process, Event
from multiprocessing.synchronize import Event as EventClass
from time import perf_counter
import argparse


def power_logging(event: EventClass, args: argparse.Namespace) -> None:
    """
    Read voltage, current and power from sys file. 

    Args:
        event: An object that manages a flag for communication among processes.
        args: Arguments from CLI.
    """
    with open(args.log_file, "w") as f:
        start_time = perf_counter()

        while not event.is_set():
            with open("", "r") as voltage_file: # Replace with sys file path 
                voltage = float(voltage_file.read())
            with open("", "r") as current_file: # Replace with sys file path 
                current = float(current_file.read())

            power = (voltage * current)
            timestamp = perf_counter() - start_time

            f.write(f"{timestamp},{voltage},{current},{power}\n")

            # For now stop after 5 seconds
            if perf_counter() - start_time > 5:
                event.set()
            # The above will be moved to the inference function later
            # where an event will be set after inference has finished. 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Power Logging for CNN Inference Cycle",
        description="Collect power usage data during inference cycles for ImageNet pretrained CNN models."
    )
    parser.add_argument(
        "--log-file-output-path",
        type=str,
        help="The path of the log file output."
    )
    args = parser.parse_args()

    event = Event()
    power_logging_process = Process(target=power_logging, args=(event, args))
    power_logging_process.start()
    power_logging_process.join()