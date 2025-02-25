"""Ecoml CLI entrypoint."""

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()
error_console = Console(stderr=True, style="bold red")
app = typer.Typer(no_args_is_help=True)

CONFIG = {"jetson_orin": {"pytorch": {"low": 5, "average": 7, "high": 10}}}


def validate_model(model_path: str):
    """Validate if the PyTorch model is a valid summary JSON file.

    Args:
        model_path: Path to PyTorch model summary

    Returns:
        A tuple of boolean if valid model and a dictionary model summary
    """
    if Path(model_path).suffix == ".json":
        try:
            with open(model_path, "r") as file:
                model_summary = json.load(file)
            return True, model_summary
        except json.JSONDecodeError:
            error_console.print("Invalid JSON file.")
            return False, None
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
    model: Annotated[str, typer.Option(help="PyTorch model summary in json format.")],
    verbose: Annotated[bool, typer.Option(help="Provide detailed summary of predictions")] = False,
    quantised: Annotated[bool, typer.Option("--quantised", help="Enable quantised prediction")] = False,
):
    """
    Predict energy estimation of a PyTorch model.

    Estimate energy consumption for PyTorch model provided using --model.

    If --verbose is used, a detailed summary of predictions is provided.
    """
    cfg = CONFIG["jetson_orin"]["pytorch"]
    success, _ = validate_model(model)

    if success:
        from ecoml.infer import run_inference, run_inference_quantised, display_metrics_table

        original_metrics = run_inference(model, power_profiles=cfg, verbose=verbose)
        if original_metrics is None or original_metrics.empty:
            error_console.print("No metrics found")
            return
        
        # Check if quantised flag passed
        if quantised:
            quantised_metrics = run_inference_quantised(model_summary_path=model, power_profiles=cfg, verbose=verbose)
            if quantised_metrics is None or quantised_metrics.empty:
                error_console.print("Quantisation failed")
                return
            
            console.print("[bold green]Quantised Results:[/]")
            display_metrics_table(quantised_metrics, model_type="Quantised")

        else:
            console.print("[bold cyan]Original Results:[/]")
            display_metrics_table(original_metrics, model_type="Original")

        # run_inference(model, power_profiles=cfg, verbose=verbose)
    else:
        error_console.print("Expected PyTorch model summary as a JSON file")

@app.command()
def compare(
    model: Annotated[str, typer.Option(help="Path to PyTorch summary in json format.")],
    verbose: Annotated[bool, typer.Option(help="Show a detailed comparison.")] = False,
):
    """
    Compares the original and quantised models
    """

    cfg = CONFIG["jetson_orin"]["pytorch"]
    success, _ = validate_model(model)

    if success:
        from ecoml.infer import run_inference, run_inference_quantised, compare_models

        original_metrics = run_inference(model, power_profiles=cfg, verbose=verbose)
        quantised_metrics = run_inference_quantised(model, power_profiles=cfg, verbose=verbose)
        compare_models(original_metrics, quantised_metrics)
    else:
        error_console.print("Expected PyTorch model summary as JSON file")


if __name__ == "__main__":
    app()
