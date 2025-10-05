"""Comprehensive command-line interface for the Agentic Code Fixer system.

This module provides a full-featured CLI that enables users to run automated code
fixing experiments, generate reports, validate configurations, and manage experiment
data through intuitive commands. It integrates rich console output for enhanced
user experience and provides comprehensive error handling.

The CLI supports the complete workflow from configuration creation through experiment
execution to result analysis and reporting.
"""

from __future__ import annotations

import asyncio
from logging import ReportGenerator
from pathlib import Path

import typer
from rich.console import Console

from coordinator import run_from_config
from core import create_default_config, load_config

app = typer.Typer(name="agentic-code-fixer", help="Automated code patch generation and evaluation")
console = Console()


@app.command()
def run(
    config_path: str = typer.Argument(..., help="Path to configuration file"),
    output_dir: str | None = typer.Option(None, help="Override output directory"),
) -> None:
    """Execute a complete automated code fixing experiment.

    Runs the full pipeline including codebase indexing, multi-agent patch generation,
    sophisticated evaluation, and comprehensive testing. Provides detailed progress
    tracking and results reporting throughout the experiment.

    Args:
        config_path: Path to YAML or JSON configuration file defining experiment parameters.
        output_dir: Optional override for the output directory specified in configuration.
    """
    try:
        config = load_config(config_path)

        if output_dir:
            config.logging.output_dir = output_dir

        console.print(f"[green]Starting experiment with config: {config_path}[/green]")

        # Run the experiment
        experiment_metadata = asyncio.run(run_from_config(config_path))

        if experiment_metadata.success:
            console.print("[green]✓ Experiment completed successfully![/green]")
            console.print(f"Winning patch: {experiment_metadata.winning_patch_id}")
            console.print(f"Duration: {experiment_metadata.total_duration_seconds:.1f}s")
        else:
            console.print(f"[red]✗ Experiment failed: {experiment_metadata.error_message}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def create_config(
    repository_path: str = typer.Argument(..., help="Path to target repository"),
    problem_description: str = typer.Argument(..., help="Description of the problem to fix"),
    model_name: str = typer.Argument(..., help="Claude model name to use (e.g., claude-sonnet-4-5-20250929)"),
    output_path: str = typer.Option("config.yaml", help="Output path for config file"),
) -> None:
    """Generate a default configuration file with sensible defaults.

    Creates a comprehensive configuration file with default settings for agents,
    evaluation methods, testing parameters, and logging options. The generated
    configuration can be customized before running experiments.

    Args:
        repository_path: Path to the target repository for automated fixing.
        problem_description: Human-readable description of the issue to address.
        model_name: Claude model name to use for agents and evaluation.
        output_path: File path where the configuration should be saved.
    """
    try:
        config = create_default_config(
            repository_path=repository_path,
            problem_description=problem_description,
            model_name=model_name,
            output_path=output_path,
        )

        console.print(f"[green]✓ Created configuration file: {output_path}[/green]")
        console.print("Edit the configuration file to customize settings before running.")

    except Exception as e:
        console.print(f"[red]Error creating config: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def report(
    experiment_dir: str = typer.Argument(..., help="Path to experiment directory"),
    format: str = typer.Option("markdown", help="Report format (markdown, json)"),
    output_file: str | None = typer.Option(None, help="Output file path"),
) -> None:
    """Generate comprehensive reports from completed experiment data.

    Creates detailed reports analyzing experiment outcomes including patch generation
    statistics, evaluation results, test outcomes, and overall performance metrics.
    Supports multiple output formats for different use cases.

    Args:
        experiment_dir: Directory containing completed experiment data and logs.
        format: Output format for the report (markdown for documentation, json for data).
        output_file: Optional specific path for the output file.
    """
    try:
        report_generator = ReportGenerator(experiment_dir)

        # Print summary to console
        report_generator.print_summary_table()
        report_generator.print_patch_details()

        # Save detailed report
        if output_file:
            saved_path = report_generator.save_report(output_file, format)
        else:
            saved_path = report_generator.save_report(format=format)

        console.print(f"[green]✓ Report saved to: {saved_path}[/green]")

    except Exception as e:
        console.print(f"[red]Error generating report: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def validate_config(
    config_path: str = typer.Argument(..., help="Path to configuration file"),
) -> None:
    """Validate configuration file syntax and parameter correctness.

    Checks configuration file format, validates all parameters, verifies file paths,
    and ensures all required settings are properly specified. Provides detailed
    feedback about any configuration issues.

    Args:
        config_path: Path to the configuration file to validate.
    """
    try:
        config = load_config(config_path)
        console.print("[green]✓ Configuration file is valid[/green]")

        # Print config summary
        console.print(f"Repository: {config.repository_path}")
        console.print(f"Agents: {len(config.agents)}")
        console.print(f"Target patches: {config.num_patch_candidates}")
        console.print(f"Evaluation method: {config.evaluation.method}")

    except Exception as e:
        console.print(f"[red]Configuration validation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_experiments(
    output_dir: str = typer.Option("./experiments", help="Experiments directory"),
) -> None:
    """Display all experiments found in the specified output directory.

    Scans the experiments directory and lists all completed and incomplete
    experiments with their status indicators. Helps users navigate and
    manage their experiment history.

    Args:
        output_dir: Directory to scan for experiment data.
    """
    try:
        experiments_path = Path(output_dir)

        if not experiments_path.exists():
            console.print(f"[yellow]No experiments directory found: {output_dir}[/yellow]")
            return

        experiments = [d for d in experiments_path.iterdir() if d.is_dir()]

        if not experiments:
            console.print(f"[yellow]No experiments found in: {output_dir}[/yellow]")
            return

        console.print(f"[bold]Experiments in {output_dir}:[/bold]")
        for exp_dir in sorted(experiments):
            experiment_file = exp_dir / "experiment.json"
            if experiment_file.exists():
                console.print(f"  [green]✓[/green] {exp_dir.name}")
            else:
                console.print(f"  [red]✗[/red] {exp_dir.name} (incomplete)")

    except Exception as e:
        console.print(f"[red]Error listing experiments: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def baseline_test(
    repository_path: str = typer.Argument(..., help="Path to repository"),
    test_command: str = typer.Option("pytest", help="Test command to run"),
) -> None:
    """Execute baseline tests on the unmodified repository.

    Runs the specified test suite on the original codebase to establish baseline
    performance and identify existing test failures. This helps understand the
    current state before applying any patches.

    Args:
        repository_path: Path to the repository to test.
        test_command: Command to execute for running tests.
    """
    try:
        from core import TestingConfig
        from patching import PatchApplicator

        testing_config = TestingConfig(test_command=test_command)
        applicator = PatchApplicator(testing_config)

        console.print("[blue]Running baseline tests...[/blue]")
        result = applicator.run_tests(repository_path)

        if result.passed:
            console.print(f"[green]✓ Tests passed ({result.duration_seconds:.1f}s)[/green]")
        else:
            console.print(f"[red]✗ Tests failed ({result.duration_seconds:.1f}s)[/red]")
            if result.failed_tests:
                console.print(f"Failed tests: {', '.join(result.failed_tests[:5])}")

    except Exception as e:
        console.print(f"[red]Error running baseline tests: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Display version information for the Agentic Code Fixer system.
    """
    from __init__ import __version__
    console.print(f"Agentic Code Fixer v{__version__}")


def main() -> None:
    """Main entry point for the command-line interface.

    Initializes the Typer application and processes command-line arguments
    to execute the appropriate commands with proper error handling.
    """
    app()


if __name__ == "__main__":
    main()