"""Script for collecting power usage during the inference cycle of a given CNN model.."""

from multiprocessing import Process, Event
from multiprocessing.synchronize import Event as EventClass
from time import time
import argparse
from pathlib import Path
from datetime import datetime
import torch
from model import prepare_inference
from tqdm import tqdm


def power_logging(event: EventClass, args: argparse.Namespace) -> None:
    """
    Read voltage, current and power from sys file.

    Args:
        event: An object that manages a flag for communication among processes.
        args: Arguments from CLI.
    """
    Path(args.result_dir).mkdir(exist_ok=True, parents=True)

    logs = []
    start_time = time()  # Current Unix timestamp

    while not event.is_set():
        with open("/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/in1_input", "r") as vdd_in: # vdd_in is global power consumption of the board
            mW = float(vdd_in.read())

        current_time = datetime.now().strftime("%H:%M:%S.%f")  # Time with seconds and microseconds
        logs.append(f"{current_time},{mW}\n")  # Log the time and power

        # For now stop after 5 seconds
        if time() - start_time > 5:
            event.set()
        # The above will be moved to the inference function later
        # where an event will be set after inference has finished.


    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f"{args.result_dir}/power_log_{current_dt}.log", "w") as f:
        f.writelines(logs)


def inference_process(
    event: EventClass,
    args: argparse.Namespace,
    trt_model,
    input_data
):
    print(f"Start timing inference cycles.")
    torch.cuda.synchronize()
    # Recorded in milliseconds
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.runs)]

    with torch.no_grad():
        for i in tqdm(range(args.runs)):
            start_events[i].record()
            _ = model(input_data)
            end_events[i].record()
        torch.cuda.synchronize()
    
    timings = [s.elapsed_time(e) * 1.0e-3 for s, e in zip(start_events, end_events)]

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

    model, input_data = prepare_inference(args)

    event = Event()
    power_logging_process = Process(target=power_logging, args=(event, args))
    power_logging_process.start()
    power_logging_process.join()
