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

    while not event.is_set():
        with open("/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/in1_input", "r") as vdd_in: # vdd_in is global power consumption of the board
            mW = float(vdd_in.read())

        current_time = datetime.now().strftime("%H:%M:%S.%f")  # Time with seconds and microseconds
        logs.append(f"{current_time},{mW}\n")  # Log the time and power

    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f"{args.result_dir}/power_log_{current_dt}.log", "w") as f:
        f.writelines(logs)


def layer_hook(layer_name, layer, input, output):
    current_time = datetime.now().strftime("%H:%M:%S.%f")
    print(f"Layer: {layer_name}, Time: {current_time}")
    

def register_hooks(model):
    hooks = []
    for name, layer in model.named_modules():
        hook = layer.register_forward_hook(lambda l, i, o: layer_hook(name, l, i, o))
        hooks.append(hook)
    return hooks


def inference(event, args, model, input_data):
    print(f"Start timing inference cycles.")
    torch.cuda.synchronize()

    # Register hooks for layer timing
    hooks = register_hooks(model)

    with torch.no_grad():
        for i in tqdm(range(args.runs)):
            _ = model(input_data)
        torch.cuda.synchronize()

    # Remove hooks after inference
    for hook in hooks:
        hook.remove()

    event.set()

# def inference_process(
#     event: EventClass,
#     args: argparse.Namespace,
#     model,
#     input_data
# ):
#     print(f"Start timing inference cycles.")
#     torch.cuda.synchronize()
#     # Recorded in milliseconds
#     start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.runs)]
#     end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.runs)]

#     with torch.no_grad():
#         for i in tqdm(range(args.runs)):
#             start_events[i].record()
#             _ = model(input_data)
#             end_events[i].record()
#         torch.cuda.synchronize()
    
#     event.set()
    
#     timings = [s.elapsed_time(e) * 1.0e-3 for s, e in zip(start_events, end_events)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Power Logging for CNN Inference Cycle",
        description="Collect power usage data during inference cycles for ImageNet pretrained CNN models."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="alexnet",
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
        "--runs",
        type=int,
        default=100,
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

    model, input_data = prepare_inference(args)

    event = Event()
    power_logging_process = Process(target=power_logging, args=(event, args))
    power_logging_process.start()

    inference_process = Process(target=inference, args=(event, args, model, input_data))
    inference_process.start()

    power_logging_process.join()
    inference_process.join()
