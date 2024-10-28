"""A script to measure the power used when the device is idling.

Before running this script, make sure that the Jetson has been set up max setting.
The same setting that will be used for inference.

To maximize Jetson Orin performance and fan speed:
    sudo /usr/bin/jetson_clocks --fan
"""
from pathlib import Path
import argparse
from datetime import datetime


import time
from datetime import datetime
from pathlib import Path
import argparse


def power_logging() -> list[tuple]:
    """
    Measure voltage and current from the system file for a duration of 1 minute.

    Returns:
        A list of instantaneous voltage and current readings captured during the measurement period.
    """
    logs = []
    start_time = time.time()  # Record the start time

    while (time.time() - start_time) < 60:  # Run for 1 minute
        # Read voltage and current from sys file
        with open("/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/in1_input", "r") as voltage:
            mV = float(voltage.read())
        with open("/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/curr1_input", "r") as current:
            mC = float(current.read())
 
        logs.append((mV, mC))
    
    return logs


def compute_average_idling_power(args: argparse.Namespace, logs: list[tuple]) -> None:
    """
    Compute and log the average idling power based on voltage and current readings.

    Args:
        args: Command-line arguments, including the output directory for storing the log file.
        logs: A list of tuples containing instantaneous voltage (mV) and current (mA) readings 
              recorded during the measurement period.

    Writes:
        A log file in the specified output directory containing the computed average idling power in mW.
    """
    mWs = [mV * mC for mV, mC in logs]
    average_mW = sum(mWs) / len(mWs)

    # Ensure the results directory exists
    Path(args.result_dir).mkdir(exist_ok=True, parents=True)
    
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f"{args.result_dir}/idling_power_log_{current_dt}.log", "w") as f:
        f.writelines(f"The average idling power measured: {average_mW} mW")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Power Logging For Idling",
        description="Collect power usage data when Jetson is idling."
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="results",
        help="The directory to save the log result."
    )
    args = parser.parse_args()
    logs = power_logging()
    compute_average_idling_power(args, logs)
