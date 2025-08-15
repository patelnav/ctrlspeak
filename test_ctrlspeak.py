#!/usr/bin/env python3
"""
ctrlSPEAK Test - A script for testing transcription with detailed logging.
"""
import torch
import sys
import os
import time
import logging
from models.factory import ModelFactory
from cli import parse_args_only
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ctrlspeak-test")
console = Console()


def main():
    """Main entry point"""
    args = parse_args_only()

    # Ensure the file argument is provided
    if not args.file:
        logger.error("The --file argument is required for testing.")
        sys.exit(1)

    # Check if file exists
    if not os.path.exists(args.file):
        logger.error(f"Audio file not found: {args.file}")
        sys.exit(1)

    # Set verbosity
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Print PyTorch info in a table
    table = Table(title="System Information", show_header=True, header_style="bold magenta")
    table.add_column("Library", style="dim", width=12)
    table.add_column("Version / Status")
    table.add_row("PyTorch", torch.__version__)
    table.add_row("CUDA", "[green]Available[/green]" if torch.cuda.is_available() else "[red]Not Available[/red]")
    table.add_row("MPS", "[green]Available[/green]" if torch.backends.mps.is_available() else "[red]Not Available[/red]")
    table.add_row("MPS Built", "[green]Yes[/green]" if torch.backends.mps.is_built() else "[red]No[/red]")
    console.print(table)

    # Enable MPS (Metal) acceleration if available
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    console.print(f"Using device: [bold cyan]{device}[/bold cyan]")

    # Resolve model alias
    resolved_model_type = ModelFactory.resolve_model_alias(args.model)
    console.print(f"Selected model (alias): [cyan]{args.model}[/cyan] -> Resolved: [cyan]{resolved_model_type}[/cyan]")

    # Load model
    with console.status(f"[bold green]Loading {resolved_model_type} model...", spinner="dots") as status:
        start_time = time.time()
        model = ModelFactory.get_model(model_type=resolved_model_type, device=device, verbose=args.debug)
        model.load_model()
        load_time = time.time() - start_time

    # Transcribe audio
    with console.status(f"[bold green]Transcribing {args.file}...", spinner="dots") as status:
        start_time = time.time()
        result = model.transcribe_batch([args.file])
        end_time = time.time()
    transcribe_time = end_time - start_time

    # Print results
    if result and result[0]:
        transcription = result[0]
        console.print(Panel(f"[bold green]{transcription}[/bold green]", title="Transcription", border_style="green"))
    else:
        console.print(Panel("[bold red]No transcription result[/bold red]", title="Error", border_style="red"))

    # Print performance metrics
    perf_table = Table(title="Performance Metrics", show_header=True, header_style="bold cyan")
    perf_table.add_column("Task", style="dim", width=20)
    perf_table.add_column("Time (seconds)", style="bold yellow")
    perf_table.add_row("Model Loading", f"{load_time:.2f}")
    perf_table.add_row("Transcription", f"{transcribe_time:.2f}")
    console.print(perf_table)


if __name__ == "__main__":
    main() 