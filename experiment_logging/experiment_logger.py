"""Comprehensive experiment logging and progress tracking for automated code fixing.

This module provides sophisticated logging capabilities that capture every aspect
of patch generation experiments, from initial configuration through final results.
It combines structured data logging with rich console output to provide both
human-readable progress tracking and machine-readable experiment data.

The logger supports multiple output formats, detailed progress visualization,
and comprehensive experiment metadata collection for analysis and reproducibility.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from core.config import LoggingConfig
from core.types import (
    EvaluationResult,
    ExperimentMetadata,
    PatchCandidate,
    TestResult,
)

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Advanced logging system providing comprehensive experiment tracking and visualization.

    This class manages all aspects of experiment logging including structured data
    collection, rich console output, file-based persistence, and progress tracking.
    It captures detailed information about patch generation, evaluation, testing,
    and outcomes for analysis and reproducibility.

    The logger provides both human-friendly console output with progress indicators
    and structured JSON logging for automated analysis. It supports configurable
    output levels and selective data capture based on logging configuration.

    Attributes:
        config: Logging configuration controlling output behavior and data capture.
        experiment_id: Unique identifier for the current experiment session.
        output_dir: Base directory for all experiment output files.
        experiment_dir: Specific directory for this experiment's data and logs.
        console: Rich console instance for formatted output (if enabled).
        experiment_data: In-memory storage for all experiment data and metrics.
    """

    def __init__(self, config: LoggingConfig, experiment_id: str) -> None:
        """Initialize the experiment logger with configuration and unique session ID.

        Sets up directory structure, console output, file logging, and initializes
        data collection structures for comprehensive experiment tracking.

        Args:
            config: LoggingConfig specifying output behavior, file locations,
                and data capture preferences.
            experiment_id: Unique identifier for this experiment session.
        """
        self.config = config
        self.experiment_id = experiment_id

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.experiment_dir = self.output_dir / experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Setup console for rich output
        self.console = Console() if config.console_output else None

        # Initialize log files
        self._setup_logging()

        # Storage for experiment data
        self.experiment_data: dict[str, Any] = {
            "experiment_id": experiment_id,
            "start_time": datetime.now().isoformat(),
            "patches": [],
            "evaluations": [],
            "test_results": [],
            "metadata": {},
        }

        logger.info(f"Initialized experiment logger for {experiment_id}")

    def _setup_logging(self) -> None:
        """Configure file-based logging with appropriate handlers and formatters.

        Sets up structured logging to capture all log messages during the experiment
        with proper formatting and appropriate log levels for detailed analysis.
        """
        log_file = self.experiment_dir / self.config.log_file

        # Configure file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, self.config.level.upper()))

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Add handler to root logger and set root logger level
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.setLevel(getattr(logging, self.config.level.upper()))
        
        # Also ensure our specific loggers are properly configured
        specific_loggers = [
            'patching.patch_applicator',
            'coordinator', 
            'experiment_logging.experiment_logger'
        ]
        
        for logger_name in specific_loggers:
            specific_logger = logging.getLogger(logger_name)
            specific_logger.setLevel(getattr(logging, self.config.level.upper()))

    def log_experiment_start(self, metadata: ExperimentMetadata) -> None:
        """Log the beginning of an experiment with comprehensive metadata.

        Records experiment configuration, displays startup information to console,
        and initializes tracking structures for the experiment session.

        Args:
            metadata: ExperimentMetadata containing configuration details,
                repository information, and experiment parameters.
        """
        self.experiment_data["metadata"] = metadata.model_dump()

        if self.console:
            panel = Panel(
                f"[bold green]Experiment Started[/bold green]\n"
                f"ID: {metadata.experiment_id}\n"
                f"Repository: {metadata.repository_path}\n"
                f"Problem: {metadata.problem_description}\n"
                f"Agents: {metadata.num_agents}\n"
                f"Target Patches: {metadata.num_patches_generated}",
                title="Agentic Code Fixer",
                border_style="green",
            )
            self.console.print(panel)

        logger.info(f"Experiment {metadata.experiment_id} started")

    def log_patch_generation_start(self, num_agents: int) -> None:
        """Log the beginning of the patch generation phase.

        Announces the start of parallel agent execution and provides visibility
        into the number of agents that will be working on patch generation.

        Args:
            num_agents: Number of AI agents participating in patch generation.
        """
        if self.console:
            self.console.print(
                f"[bold blue]Starting patch generation with {num_agents} agents...[/bold blue]"
            )

        logger.info(f"Starting patch generation with {num_agents} agents")

    def log_patch_generated(self, patch: PatchCandidate) -> None:
        """Log the successful generation of a patch candidate.

        Records patch details, displays progress information, and optionally
        stores the complete patch data based on configuration settings.

        Args:
            patch: PatchCandidate object containing the generated solution
                and metadata about its creation.
        """
        if self.config.save_patches:
            self.experiment_data["patches"].append(patch.model_dump())

        if self.console:
            self.console.print(
                f"✓ Patch generated by {patch.agent_id} "
                f"(confidence: {patch.confidence_score:.2f})"
            )

        logger.info(f"Patch {patch.id} generated by {patch.agent_id}")

    def log_patch_generation_complete(self, patches: list[PatchCandidate]) -> None:
        """Log the completion of patch generation with comprehensive summary.

        Displays a detailed table of all generated patches including agent sources,
        confidence scores, and descriptions. Provides clear visibility into the
        diversity and quality of generated solutions.

        Args:
            patches: Complete list of PatchCandidate objects generated during
                the experiment.
        """
        if self.console:
            table = Table(title="Generated Patches")
            table.add_column("Agent", style="cyan")
            table.add_column("File", style="magenta")
            table.add_column("Confidence", justify="right", style="green")
            table.add_column("Description", style="yellow")

            for patch in patches:
                table.add_row(
                    patch.agent_id,
                    Path(patch.file_path).name,
                    f"{patch.confidence_score:.2f}",
                    (
                        patch.description[:50] + "..."
                        if len(patch.description) > 50
                        else patch.description
                    ),
                )

            self.console.print(table)

        logger.info(f"Patch generation completed: {len(patches)} patches generated")

    def log_evaluation_start(self, num_patches: int, method: str) -> None:
        """Log the beginning of the patch evaluation phase.

        Announces the start of patch comparison and evaluation, providing
        visibility into the evaluation methodology and scope.

        Args:
            num_patches: Number of patch candidates to be evaluated.
            method: Evaluation method being used (e.g., 'pairwise', 'tournament').
        """
        if self.console:
            self.console.print(
                f"[bold blue]Starting {method} evaluation of {num_patches} patches...[/bold blue]"
            )

        logger.info(f"Starting {method} evaluation of {num_patches} patches")

    def log_evaluation_result(self, result: EvaluationResult) -> None:
        """Log the outcome of a single patch comparison.

        Records detailed evaluation results including winner determination,
        confidence scores, and reasoning. Optionally stores complete evaluation
        data based on configuration settings.

        Args:
            result: EvaluationResult containing comparison outcome, confidence
                scores, and detailed reasoning.
        """
        if self.config.save_evaluations:
            self.experiment_data["evaluations"].append(result.model_dump())

        logger.debug(
            f"Evaluation: {result.patch_a_id} vs {result.patch_b_id} -> {result.winner_id}"
        )

    def log_evaluation_complete(
        self, results: list[EvaluationResult], winner: PatchCandidate | None
    ) -> None:
        """Log the completion of evaluation with winner announcement.

        Displays evaluation summary and highlights the winning patch candidate
        with detailed information about its characteristics and selection rationale.

        Args:
            results: Complete list of EvaluationResult objects from all comparisons.
            winner: The PatchCandidate selected as the best solution, or None
                if no clear winner was determined.
        """
        if self.console:
            self.console.print(
                f"[green]✓ Evaluation completed: {len(results)} comparisons[/green]"
            )

            if winner:
                panel = Panel(
                    f"[bold green]Winner Selected[/bold green]\n"
                    f"Patch ID: {winner.id}\n"
                    f"Agent: {winner.agent_id}\n"
                    f"Confidence: {winner.confidence_score:.2f}\n"
                    f"Description: {winner.description}",
                    title="Best Patch",
                    border_style="green",
                )
                self.console.print(panel)

        logger.info(
            f"Evaluation completed: {len(results)} comparisons, winner: {winner.id if winner else 'None'}"
        )

    def log_test_start(self, patch_id: str) -> None:
        """Log the beginning of patch testing and validation.

        Announces the start of test execution for the selected patch,
        providing visibility into the validation process.

        Args:
            patch_id: Unique identifier of the patch being tested.
        """
        if self.console:
            self.console.print(f"[bold blue]Testing patch {patch_id}...[/bold blue]")

        logger.info(f"Starting test of patch {patch_id}")

    def log_test_result(self, result: TestResult, patch_id: str) -> None:
        """Log the outcome of patch testing with detailed results.

        Records comprehensive test results including pass/fail status, execution
        time, and failure details. Provides clear visibility into patch
        effectiveness and any regression issues.

        Args:
            result: TestResult containing test outcomes, timing, and failure details.
            patch_id: Unique identifier of the tested patch.
        """
        if self.config.save_test_results:
            test_data = result.model_dump()
            test_data["patch_id"] = patch_id
            self.experiment_data["test_results"].append(test_data)

        status = "PASSED" if result.passed else "FAILED"
        color = "green" if result.passed else "red"

        if self.console:
            self.console.print(
                f"[{color}]✓ Tests {status} ({result.duration_seconds:.1f}s)[/{color}]"
            )

            if not result.passed and result.failed_tests:
                self.console.print(
                    f"[red]Failed tests: {', '.join(result.failed_tests[:3])}[/red]"
                )

        logger.info(
            f"Test result for patch {patch_id}: {status} ({result.duration_seconds:.1f}s)"
        )

    def log_experiment_complete(self, metadata: ExperimentMetadata) -> None:
        """Log the completion of the entire experiment with comprehensive results.

        Finalizes experiment data collection, saves all gathered information to
        persistent storage, and displays a comprehensive summary of outcomes
        including success status and performance metrics.

        Args:
            metadata: ExperimentMetadata containing final results, timing
                information, and success indicators.
        """
        self.experiment_data["end_time"] = datetime.now().isoformat()
        self.experiment_data["duration_seconds"] = metadata.total_duration_seconds
        self.experiment_data["success"] = metadata.success
        self.experiment_data["winning_patch_id"] = metadata.winning_patch_id
        self.experiment_data["error_message"] = metadata.error_message

        # Save experiment data to file
        experiment_file = self.experiment_dir / "experiment.json"
        with open(experiment_file, "w", encoding="utf-8") as f:
            json.dump(self.experiment_data, f, indent=2, default=str)

        if self.console:
            status = "SUCCESS" if metadata.success else "FAILED"
            color = "green" if metadata.success else "red"

            panel = Panel(
                f"[bold {color}]Experiment {status}[/bold {color}]\n"
                f"Duration: {metadata.total_duration_seconds:.1f}s\n"
                f"Winning Patch: {metadata.winning_patch_id or 'None'}\n"
                f"Results saved to: {self.experiment_dir}",
                title="Experiment Complete",
                border_style=color,
            )
            self.console.print(panel)

        logger.info(f"Experiment {self.experiment_id} completed: {status}")

    def log_error(self, message: str, exception: Exception | None = None) -> None:
        """Log error conditions encountered during experiment execution.

        Records error messages and optional exception details to both console
        and file logs for debugging and analysis purposes.

        Args:
            message: Human-readable error description.
            exception: Optional Exception object providing additional context.
        """
        if self.console:
            self.console.print(f"[bold red]ERROR: {message}[/bold red]")

        if exception:
            logger.error(f"{message}: {exception}")
        else:
            logger.error(message)

    def log_warning(self, message: str) -> None:
        """Log warning conditions that don't stop execution but require attention.

        Records warning messages to both console and file logs to highlight
        potential issues or suboptimal conditions.

        Args:
            message: Human-readable warning description.
        """
        if self.console:
            self.console.print(f"[yellow]WARNING: {message}[/yellow]")

        logger.warning(message)

    def log_info(self, message: str) -> None:
        """Log informational messages about experiment progress.

        Records general information messages to both console and file logs
        for progress tracking and debugging purposes.

        Args:
            message: Human-readable informational message.
        """
        if self.console:
            self.console.print(f"[blue]INFO: {message}[/blue]")

        logger.info(message)

    def create_progress_context(self, description: str):
        """Create a rich progress indicator for long-running operations.

        Provides visual progress feedback for operations that may take significant
        time to complete, improving user experience and operation transparency.

        Args:
            description: Description of the operation being tracked.

        Returns:
            Progress context manager for visual progress indication, or dummy
            context if console output is disabled.
        """
        """Create a progress context for long-running operations."""
        if self.console:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            )
        else:
            # Return a dummy context manager
            class DummyProgress:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def add_task(self, description, total=None):
                    return 0

                def update(self, task_id, advance=1):
                    pass

            return DummyProgress()

    def get_experiment_data(self) -> dict[str, Any]:
        """Retrieve a copy of all collected experiment data.

        Returns:
            Dictionary containing complete experiment data including patches,
            evaluations, test results, and metadata.
        """
        return self.experiment_data.copy()

    def export_logs(self, format: str = "json") -> Path:
        """Export comprehensive experiment logs to a structured file format.

        Creates a complete export of all experiment data in the specified format
        for external analysis, reporting, or archival purposes.

        Args:
            format: Export format (currently supports 'json').

        Returns:
            Path to the created export file.

        Raises:
            ValueError: If the specified format is not supported.
        """
        if format == "json":
            output_file = self.experiment_dir / "experiment_export.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.experiment_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Exported experiment logs to {output_file}")
        return output_file
