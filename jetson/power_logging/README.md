# Measuring Power

There are two processes for measuring the power for an inference cycle:
1. The power logging process
2. The inference process (WIP)

The final output would be a graph of power consumption per layer.

Tested the power logging process on the Jetson Orin Nano.

## Getting Started 

```bash
uv venv --python 3.11
source .venv/bin/activate
```

### Running the power logger

```bash
uv run python measure_power.py
```

By default the results will be output into the results folder.