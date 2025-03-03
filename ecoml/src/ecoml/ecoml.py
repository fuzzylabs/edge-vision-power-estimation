"""Ecoml CLI entrypoint."""

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

CONFIG = {"jetson_orin": {"pytorch": {"low": 5, "average": 7, "high": 10}}}

console = Console()
error_console = Console(stderr=True, style="bold red")
app = typer.Typer(no_args_is_help=True)


def validate_model(model_path: str):
    """Validate if the PyTorch model is a valid summary JSON file or PyTorch Model.

    Args:
        model_path: Path to PyTorch model summary

    Returns:
        A tuple of boolean if valid model and a dictionary model summary
    """
    path = Path(model_path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        try:
            with open(path, "r") as file:
                _ = json.load(file)
            return True, "json"
        except json.JSONDecodeError:
            error_console.print("Invalid JSON file.")
            return False, None

    if suffix in [".pt", ".pth"]:
        return True, "pt"
    
    return False, None


def display_config_table(cfg: dict[str, int]) -> None:
    """Display a table of power profiles.

    Args:
        cfg: Dictionary of data containing power profiles.
    """
    table = Table(title="PyTorch Power profiles")
    table.add_column("Power profile", justify="center", style="cyan", no_wrap=True)
    table.add_column("Watt", justify="center", style="green")

    for key, value in cfg.items():
        table.add_row(key.capitalize() + " Bound", str(value))
    console.print(table)


@app.command()
def config():
    """Power configuration used for energy estimation."""
    cfg = CONFIG["jetson_orin"]["pytorch"]
    display_config_table(cfg)


@app.command()
def predict(
    model: Annotated[str, typer.Option(help="PyTorch model (JSON or .pt file)")],
    verbose: Annotated[bool, typer.Option(help="Detailed Summary")] = False,
):
    """
    Predict energy estimation of a PyTorch model.

    Estimate energy consumption for PyTorch model provided using --model.

    If --verbose is used, a detailed summary of predictions is provided.
    """
    cfg = CONFIG["jetson_orin"]["pytorch"]

    success, model_type = validate_model(model)
    if not success:
        error_console.print("Invalid file type. Must be a .json or .pt")
        raise typer.Exit(code=1)
    
    # Import relevant functions
    from ecoml.infer import(
        run_inference,
        run_inference_pt,
        display_latency_table,
        display_metrics_table,
        display_runtime_table
    )

    if model_type == "json":
        runtime_predictions = run_inference(Path(model), power_profiles=cfg, verbose=verbose)
    else:
        runtime_predictions = run_inference_pt(Path(model), power_profiles=cfg, verbose=verbose)

    # If it is an empty dict then throw an error
    if not runtime_predictions:
        error_console.print("Inference failed. No results were returned")
        raise typer.Exit(code=1)
    
    # Take out data from the dict
    # layer_df = results["layer_data"]
    # metrics_df = results["metrics_data"]

    # Display table
    if verbose:
        display_latency_table(runtime_predictions)

    display_metrics_table(runtime_predictions, power_profiles=cfg) # Get this working again -> comparison after 
    display_runtime_table(runtime_predictions)

@app.command()
def compare(
    model1: Annotated[str, typer.Option(help="PyTorch model summary in json format.")],
    model2: Annotated[str, typer.Option(help="Second PyTorch model summary in json format")],
    verbose: bool = False,
):
    """
    Compare two models and visualise improvement.
    """
    from ecoml.infer import(
        run_inference,
        display_comparison_table,
        display_latency_table,
        display_metrics_table,
        display_runtime_table
    )

    cfg = CONFIG["jetson_orin"]["pytorch"]

    results_baseline = run_inference(Path(model1), cfg, verbose=verbose)

    if model2 is None:
        display_latency_table(results_baseline)
        display_runtime_table(results_baseline)
        display_metrics_table(results_baseline, cfg)
        return

    results_compare = run_inference(Path(model2), cfg, verbose=verbose)

    display_comparison_table(results_baseline, results_compare, cfg)

    return

if __name__ == "__main__":
    app()
