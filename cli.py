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
from pathlib import Path

import typer
from rich.console import Console

from coordinator import run_from_config_with_overrides
from core import create_default_config, load_config
from core.role_manager import RoleManager
from experiment_logging import ReportGenerator

app = typer.Typer(
    name="agentic-code-fixer", help="Automated code patch generation and evaluation"
)
console = Console()


@app.command()
def run(
    config_path: str = typer.Argument(..., help="Path to configuration file"),
    input: str = typer.Option(
        ..., "--input", help="Description of the issue to fix (required)"
    ),
    output_dir: str | None = typer.Option(None, help="Override output directory"),
    context: list[str] | None = typer.Option(
        None,
        "--context",
        help="Additional context files to include along with vectordb data",
    ),
    exclude_patterns: list[str] | None = typer.Option(
        None,
        "--exclude-patterns",
        help="Additional file patterns to exclude during indexing (e.g., '*.log', 'temp_*')",
    ),
) -> None:
    """Execute a complete automated code fixing experiment.

    Runs the full pipeline including codebase indexing, multi-agent patch generation,
    sophisticated evaluation, and comprehensive testing. Provides detailed progress
    tracking and results reporting throughout the experiment.

    Args:
        config_path: Path to YAML configuration file defining experiment parameters.
        input: Description of the issue/problem to fix.
        output_dir: Optional override for the output directory specified in configuration.
        context: Optional list of additional context files to include with vectordb data.
        exclude_patterns: Optional list of additional file patterns to exclude during indexing.
    """
    try:
        config = load_config(config_path)

        # Set the problem description from command line input
        config.problem_description = input

        # Add context files if provided (note: with target_files removed, context files are now indexed automatically through comprehensive repository scanning)
        if context:
            # Validate context files exist and log them for user awareness
            context_files = []
            for ctx_file in context:
                ctx_path = Path(ctx_file)
                if ctx_path.exists():
                    context_files.append(str(ctx_path.resolve()))
                    console.print(
                        f"[blue]Context file found: {ctx_file} (will be included via automatic repository indexing)[/blue]"
                    )
                else:
                    console.print(
                        f"[yellow]Warning: Context file not found: {ctx_file}[/yellow]"
                    )

        # Add additional exclude patterns if provided
        if exclude_patterns:
            # Merge CLI exclude patterns with config exclude patterns
            config.exclude_patterns.extend(exclude_patterns)
            console.print(
                f"[blue]Added exclude patterns: {', '.join(exclude_patterns)}[/blue]"
            )

        if output_dir:
            config.logging.output_dir = output_dir

        console.print(f"[green]Starting experiment with config: {config_path}[/green]")
        console.print(f"[green]Problem: {input}[/green]")
        if context:
            console.print(
                f"[green]Additional context files: {len(context_files)}[/green]"
            )
        if exclude_patterns:
            console.print(
                f"[green]Additional exclude patterns: {len(exclude_patterns)}[/green]"
            )

        # Run the experiment with modified config
        experiment_metadata = asyncio.run(run_from_config_with_overrides(config))

        if experiment_metadata.success:
            console.print("[green]✓ Experiment completed successfully![/green]")
            console.print(f"Winning patch: {experiment_metadata.winning_patch_id}")
            console.print(
                f"Duration: {experiment_metadata.total_duration_seconds:.1f}s"
            )
        else:
            console.print(
                f"[red]✗ Experiment failed: {experiment_metadata.error_message}[/red]"
            )
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def create_config(
    repository_path: str = typer.Argument(..., help="Path to target repository"),
    problem_description: str = typer.Argument(
        ..., help="Description of the problem to fix"
    ),
    model_name: str = typer.Argument(
        ..., help="Claude model name to use (e.g., claude-sonnet-4-5-20250929)"
    ),
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

        # Print config summary (similar to validate_config)
        console.print(f"Repository: {config.repository_path}")
        console.print(f"Agents: {len(config.agents)}")
        console.print(f"Target candidate solutions: {config.num_candidate_solutions}")
        console.print(f"Model: {config.evaluation.model_name}")

        console.print(
            "Edit the configuration file to customize settings before running."
        )

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
        console.print(f"Target candidate solutions: {config.num_candidate_solutions}")
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
            console.print(
                f"[yellow]No experiments directory found: {output_dir}[/yellow]"
            )
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
            console.print(
                f"[green]✓ Tests passed ({result.duration_seconds:.1f}s)[/green]"
            )
        else:
            console.print(f"[red]✗ Tests failed ({result.duration_seconds:.1f}s)[/red]")
            if result.failed_tests:
                console.print(f"Failed tests: {', '.join(result.failed_tests[:5])}")

    except Exception as e:
        console.print(f"[red]Error running baseline tests: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_roles(
    roles_dir: str = typer.Option(
        "roles", help="Directory containing role definitions"
    ),
    category: str = typer.Option(None, help="Filter by category"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed role information"
    ),
) -> None:
    """List all available agent roles and their descriptions.

    This command helps users discover available roles for configuring agents.
    Each role provides specialized prompting for different types of code issues.

    Args:
        roles_dir: Directory containing role definition files.
        category: Optional category filter (e.g., security, performance).
        verbose: Show detailed role information including descriptions.
    """
    try:
        role_manager = RoleManager(roles_directory=roles_dir)

        # Get role statistics
        stats = role_manager.get_role_stats()
        console.print(f"[bold]Available Roles ({stats['total_roles']} total):[/bold]")
        console.print(f"Roles directory: {stats['roles_directory']}")

        if category:
            roles = role_manager.get_roles_by_category(category)
            if not roles:
                console.print(
                    f"[yellow]No roles found in category '{category}'[/yellow]"
                )
                return
            console.print(f"[blue]Filtered by category: {category}[/blue]")
        else:
            roles = [
                role_manager.get_role_definition(name)
                for name in role_manager.list_available_roles()
            ]
            roles = [role for role in roles if role is not None]

        # Display roles
        for role in sorted(roles, key=lambda r: r.name):
            if verbose:
                console.print(f"\n[bold green]{role.name}[/bold green]")
                console.print(f"  Description: {role.description}")
                if role.category:
                    console.print(f"  Category: {role.category}")
                if role.priority:
                    console.print(f"  Priority: {role.priority}")
                if role.tags:
                    console.print(f"  Tags: {', '.join(role.tags)}")
                console.print("  Prompt Addition:")
                # Indent the prompt addition
                for line in role.prompt_addition.split("\n"):
                    console.print(f"    {line}")
            else:
                category_str = f" ({role.category})" if role.category else ""
                console.print(
                    f"  [green]{role.name}[/green]{category_str}: {role.description}"
                )

        # Show category summary if not filtering
        if not category and stats["categories"]:
            console.print("\n[bold]Categories:[/bold]")
            for cat, count in sorted(stats["categories"].items()):
                console.print(f"  {cat}: {count} roles")

    except Exception as e:
        console.print(f"[red]Failed to list roles: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Display version information for the Agentic Code Fixer system."""
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
