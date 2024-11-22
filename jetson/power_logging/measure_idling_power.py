"""A script to measure the power used when the device is idling.

Before running this script, make sure that the Jetson has been set up max setting.
The same setting that will be used for inference.

To maximize Jetson Orin performance and fan speed:
    sudo /usr/bin/jetson_clocks --fan
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path


def power_logging(idle_duration: int) -> list[tuple]:
    """
    Measure voltage and current from the system file.

    Args:
        idle_duration: Time to be spent collecting power values on Jeston.

    Returns:
        A list of instantaneous voltage and current readings captured during the measurement period.
    """
    logs = []
    start_time = time.time()  # Record the start time

    print(f"Logging idling power usage for {idle_duration} seconds ...")
    while (time.time() - start_time) < idle_duration:
        # Read voltage and current from sys file
        with open(
            "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/in1_input", "r"
        ) as voltage:
            mV = float(voltage.read())
        with open(
            "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/curr1_input", "r"
        ) as current:
            mC = float(current.read())

        logs.append((mV, mC))
    print("Power logging complete.")

    return logs


def compute_average_idling_power(result_dir: Path, logs: list[tuple]) -> None:
    """
    Compute and log the average idling power based on voltage and current readings.

    Args:
        result_dir: Path to directory to save idle power in a json file.
        logs: A list of tuples containing instantaneous voltage (mV) and current (mA) readings
            recorded during the measurement period.

    Writes:
        A log file in the specified output directory containing the computed average idling power in mW.
    """
    print("Computing average idling power.")
    mWs = [mV * mC for mV, mC in logs]
    average_mW = sum(mWs) / len(mWs)
    print(f"Average idling power: {average_mW} muW")

    # Ensure the results directory exists
    result_dir.mkdir(exist_ok=True, parents=True)
    json_file_path = f"{result_dir}/idling_power.json"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(json_file_path, "w") as fp:
        json.dump({"timestamp": timestamp, "avg_idle_power": average_mW}, fp, indent=4)

    print(f"Log file created at {json_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Power Logging For Idling",
        description="Collect power usage data when Jetson is idling.",
    )
    parser.add_argument(
        "--idle-duration",
        type=int,
        default=60,
        help="Duration (in seconds) to measure power usage during idle state. Default is 60 seconds.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="results",
        help="The directory to save the log result.",
    )
    args = parser.parse_args()

    logs = power_logging(args.idle_duration)
    compute_average_idling_power(Path(args.result_dir), logs)
