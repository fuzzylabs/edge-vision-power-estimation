"""Ecoml CLI entrypoint."""

import json
import typer
from pathlib import Path
from typing import Annotated
from rich.console import Console
from rich.table import Table
from ecoml.infer import (
    run_inference,
    display_latency_table,
    display_metrics_table,
    display_runtime_table,
    display_comparison_table
)


CONFIG = {"jetson_orin": {"pytorch": {"low": 5, "average": 7, "high": 10}}}

console = Console()
error_console = Console(stderr=True, style="bold red")
app = typer.Typer(no_args_is_help=True)


def validate_model(model_path: str):
    """Validate if the PyTorch model is a valid summary JSON file.

    Args:
        model_path: Path to PyTorch model summary

    Returns:
        A tuple of boolean if valid model and a dictionary model summary
    """
    path = Path(model_path)
    if path.suffix != ".json":
        raise typer.BadParameter("Expected a JSON File")
    try:
        with path.open("r") as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        raise typer.BadParameter("Cannot read file")


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
    model: str = typer.Option(..., help="PyTorch model summary in json format."),
    verbose: bool = typer.Option(False, help="Detailed Summary"),
):
    """
    Predict energy estimation of a PyTorch model.

    Estimate energy consumption for PyTorch model provided using --model.

    If --verbose is used, a detailed summary of predictions is provided.
    """
    cfg = CONFIG["jetson_orin"]["pytorch"]

    try:
        validate_model(model)
    except typer.BadParameter as e:
        error_console.print(e)
        raise typer.Exit(code=1)
    
    # Run the inference function that returns the dictionary
    runtime_predictions = run_inference(Path(model), power_profiles=cfg, verbose=verbose)

    # If it is an empty dict then throw an error
    if not runtime_predictions:
        error_console.print("Inference failed. No results were returned")
        raise typer.Exit(code=1)

    # Display table
    if verbose:
        display_latency_table(runtime_predictions)

    display_metrics_table(runtime_predictions, power_profiles=cfg) # Get this working again -> comparison after 
    display_runtime_table(runtime_predictions)

@app.command()
def compare(
    model1: str = typer.Option(..., help="PyTorch model summary in json format."),
    model2: str = typer.Option(None, help="Second PyTorch model summary in json format"),
    verbose: bool = typer.Option(False, help="Show detailed comparison"),
):
    """
    Provide comparison between two models.
    """
    cfg = CONFIG["jetson_orin"]["pytorch"]

    try:
        validate_model(model1)
        results_baseline = run_inference(Path(model1), cfg, verbose=verbose)
    except typer.BadParameter as e:
        error_console.print(f"Model 1 Error: {e}")
        raise typer.Exit(code=1)
    
    if not model2:
        if verbose:
            display_latency_table(results_baseline)
        display_runtime_table(results_baseline)
        display_metrics_table(results_baseline, cfg)
        return
    
    try:
        validate_model(model2)
        results_compare = run_inference(Path(model2), cfg, verbose=verbose)
    except typer.BadParameter as e:
        error_console.print(f"Model 2 Error: {e}")
        raise typer.Exit(code=1)

    display_comparison_table(results_baseline, results_compare, cfg)

if __name__ == "__main__":
    app()
