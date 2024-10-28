# Measuring Power

To measure power consumption during an inference cycle, two processes are used similar to the profiling energy paper:
1. **Power Logging Process**: Captures power data.
2. **Inference Process**: Runs inference to measure power used by each layer.

The goal is to produce a graph of power consumption per layer. (*WIP SD-57*)

The power measurement and inference process have been tested on the Jetson Orin Nano.

## Getting Started 

### Maximise Jetson Orin Performance and Set Fan Speed

Run the following command to maximize performance and set the fan speed:

```bash
sudo /usr/bin/jetson_clocks --fan
```

### Running the Power Measurement

To use our power measurement script, run it inside the Docker image `nvcr.io/nvidia/pytorch:24.06-py3-igpu` (approx. 5 GB in size).
> **Important**: Use the specified image to avoid compatibility issues with `tensorrt` and `torch_tensorrt` versions.

Start the container with:

```bash
sudo docker run --runtime=nvidia -it -v <your_project_directory>:/home nvcr.io/nvidia/pytorch:24.06-py3-igpu
```

Since we’ve mounted our project directory to `/home`, switch to that directory before running the script:

```bash
cd /home
ls
# Your project folder should appear here, e.g., innovation-power-estimation-models
```

### Running Power Measurement Scripts

You’ll need to run two scripts for measuring power:

1. **[measure_idling_power.py](measure_idling_power.py)** - Measures the idling power of the Jetson Orin Nano. Ensure the performance settings are applied before running this. This script outputs an `idling_power_log_{timestamp}.log` file with idling power data.

2. **[measure_inference_power.py](measure_inference_power.py)** - Measures instantaneous power consumption and timestamps for each inference cycle. You can specify the number of inference cycles with the `--runs` argument.

For example, to run `measure_inference_power.py` with 10,000 inference cycles:

```bash
python measure_inference_power.py --runs 10000
```

This script generates multiple log and trace files. The two primary files of interest are:
1. `power_log_{timestamp}.log`: Logs power measurements during inference.
2. `trt.log`: Contains additional inference information.

By default, all results are saved in the `results` folder.