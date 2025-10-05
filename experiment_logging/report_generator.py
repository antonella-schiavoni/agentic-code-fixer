"""Advanced report generation system for comprehensive experiment analysis and documentation.

This module provides sophisticated reporting capabilities that transform raw experiment
data into readable, informative reports and visualizations. It supports multiple output
formats including Markdown and JSON, and provides both summary and detailed views
of experiment results.

The report generator analyzes patch generation outcomes, evaluation results, test
performance, and overall experiment success to create comprehensive documentation
suitable for analysis, sharing, and archival purposes.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table


class ReportGenerator:
    """Advanced report generation system for experiment analysis and documentation.

    This class transforms raw experiment data into comprehensive, readable reports
    that provide detailed insights into patch generation performance, evaluation
    outcomes, and overall experiment success. It supports multiple output formats
    and visualization styles for different use cases.

    The generator can create summary reports for quick overview, detailed reports
    for in-depth analysis, and statistical exports for further processing. It also
    provides rich console output for interactive exploration of results.

    Attributes:
        experiment_dir: Directory containing experiment data and output files.
        console: Rich console instance for formatted output display.
        experiment_data: Loaded experiment data including all metrics and outcomes.
    """

    def __init__(self, experiment_dir: str | Path) -> None:
        """Initialize the report generator with experiment data.

        Loads experiment data from the specified directory and prepares the
        generator for report creation and analysis.

        Args:
            experiment_dir: Path to directory containing experiment.json and
                related experiment data files.

        Raises:
            FileNotFoundError: If experiment data file is not found in the directory.
        """
        self.experiment_dir = Path(experiment_dir)
        self.console = Console()

        # Load experiment data
        self.experiment_data = self._load_experiment_data()

    def _load_experiment_data(self) -> dict[str, Any]:
        """Load and parse experiment data from the JSON data file.

        Returns:
            Dictionary containing complete experiment data including metadata,
            patches, evaluations, and test results.

        Raises:
            FileNotFoundError: If the experiment.json file is not found.
            json.JSONDecodeError: If the experiment data is malformed.
        """
        experiment_file = self.experiment_dir / "experiment.json"

        if not experiment_file.exists():
            raise FileNotFoundError(f"Experiment data not found: {experiment_file}")

        with open(experiment_file, encoding="utf-8") as f:
            return json.load(f)

    def generate_summary_report(self) -> str:
        """Generate a concise Markdown summary report of experiment outcomes.

        Creates a high-level overview report including key metrics, success status,
        patch generation statistics, evaluation results, and test outcomes.
        Suitable for quick assessment and sharing.

        Returns:
            Markdown-formatted string containing the complete summary report.
        """
        metadata = self.experiment_data.get("metadata", {})
        patches = self.experiment_data.get("patches", [])
        evaluations = self.experiment_data.get("evaluations", [])
        test_results = self.experiment_data.get("test_results", [])

        report = f"""# Agentic Code Fixer Experiment Report

## Experiment Overview
- **Experiment ID**: {metadata.get("experiment_id", "Unknown")}
- **Repository**: {metadata.get("repository_path", "Unknown")}
- **Problem**: {metadata.get("problem_description", "Unknown")}
- **Start Time**: {self.experiment_data.get("start_time", "Unknown")}
- **Duration**: {self.experiment_data.get("duration_seconds", 0):.1f} seconds
- **Status**: {"SUCCESS" if self.experiment_data.get("success", False) else "FAILED"}

## Patch Generation Results
- **Total Patches Generated**: {len(patches)}
- **Number of Agents**: {metadata.get("num_agents", 0)}
- **Target Patch Count**: {metadata.get("num_patches_generated", 0)}

### Patch Statistics by Agent
"""

        # Add patch statistics by agent
        agent_stats = {}
        for patch in patches:
            agent_id = patch.get("agent_id", "Unknown")
            if agent_id not in agent_stats:
                agent_stats[agent_id] = {
                    "count": 0,
                    "avg_confidence": 0,
                    "total_confidence": 0,
                }

            agent_stats[agent_id]["count"] += 1
            agent_stats[agent_id]["total_confidence"] += patch.get(
                "confidence_score", 0
            )

        for agent_id, stats in agent_stats.items():
            avg_confidence = (
                stats["total_confidence"] / stats["count"] if stats["count"] > 0 else 0
            )
            report += f"- **{agent_id}**: {stats['count']} patches (avg confidence: {avg_confidence:.2f})\n"

        report += f"""
## Evaluation Results
- **Total Evaluations**: {len(evaluations)}
- **Evaluation Method**: {metadata.get("evaluation_method", "Unknown")}
- **Winning Patch**: {self.experiment_data.get("winning_patch_id", "None")}

"""

        # Add evaluation statistics
        if evaluations:
            avg_confidence = sum(e.get("confidence", 0) for e in evaluations) / len(
                evaluations
            )
            report += f"- **Average Evaluation Confidence**: {avg_confidence:.2f}\n"

            # Count wins by patch
            patch_wins = {}
            for eval_result in evaluations:
                winner = eval_result.get("winner_id")
                if winner:
                    patch_wins[winner] = patch_wins.get(winner, 0) + 1

            if patch_wins:
                report += "\n### Wins by Patch\n"
                for patch_id, wins in sorted(
                    patch_wins.items(), key=lambda x: x[1], reverse=True
                ):
                    report += f"- **{patch_id}**: {wins} wins\n"

        # Add test results
        if test_results:
            report += f"""
## Test Results
- **Total Tests Run**: {len(test_results)}
"""
            passed_tests = sum(1 for t in test_results if t.get("passed", False))
            report += f"- **Tests Passed**: {passed_tests}/{len(test_results)}\n"

            if test_results:
                avg_duration = sum(
                    t.get("duration_seconds", 0) for t in test_results
                ) / len(test_results)
                report += f"- **Average Test Duration**: {avg_duration:.1f} seconds\n"

        # Add error information if experiment failed
        if not self.experiment_data.get("success", False):
            error_msg = self.experiment_data.get("error_message")
            if error_msg:
                report += f"""
## Error Information
```
{error_msg}
```
"""

        return report

    def generate_detailed_report(self) -> str:
        """Generate a comprehensive detailed report with complete experiment data.

        Creates an extensive Markdown report including the summary information
        plus detailed breakdowns of individual patches, complete evaluation
        results with reasoning, and comprehensive analysis data.

        Returns:
            Markdown-formatted string containing the complete detailed report
            with all experiment information.
        """
        summary = self.generate_summary_report()

        patches = self.experiment_data.get("patches", [])
        evaluations = self.experiment_data.get("evaluations", [])

        detailed_report = (
            summary
            + """

## Detailed Patch Information

"""
        )

        # Add detailed patch information
        for i, patch in enumerate(patches, 1):
            detailed_report += f"""### Patch {i}: {patch.get("id", "Unknown")}
- **Agent**: {patch.get("agent_id", "Unknown")}
- **File**: {patch.get("file_path", "Unknown")}
- **Lines**: {patch.get("line_start", 0)}-{patch.get("line_end", 0)}
- **Confidence**: {patch.get("confidence_score", 0):.2f}
- **Description**: {patch.get("description", "No description")}
- **Status**: {patch.get("status", "Unknown")}

```
{patch.get("content", "No content available")}
```

"""

        # Add evaluation details
        if evaluations:
            detailed_report += """
## Detailed Evaluation Results

"""
            for i, evaluation in enumerate(evaluations, 1):
                detailed_report += f"""### Evaluation {i}
- **Patch A**: {evaluation.get("patch_a_id", "Unknown")}
- **Patch B**: {evaluation.get("patch_b_id", "Unknown")}
- **Winner**: {evaluation.get("winner_id", "Unknown")}
- **Confidence**: {evaluation.get("confidence", 0):.2f}
- **Reasoning**: {evaluation.get("reasoning", "No reasoning provided")}

"""

        return detailed_report

    def print_summary_table(self) -> None:
        """Display a formatted summary table of key experiment metrics.

        Prints a rich-formatted table to the console showing essential experiment
        information including ID, duration, patch count, success status, and
        winning patch for quick visual assessment.
        """
        metadata = self.experiment_data.get("metadata", {})
        patches = self.experiment_data.get("patches", [])

        # Create summary table
        table = Table(title="Experiment Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Experiment ID", metadata.get("experiment_id", "Unknown"))
        table.add_row("Repository", str(Path(metadata.get("repository_path", "")).name))
        table.add_row(
            "Duration", f"{self.experiment_data.get('duration_seconds', 0):.1f}s"
        )
        table.add_row("Patches Generated", str(len(patches)))
        table.add_row(
            "Status",
            "SUCCESS" if self.experiment_data.get("success", False) else "FAILED",
        )
        table.add_row(
            "Winning Patch", self.experiment_data.get("winning_patch_id", "None")
        )

        self.console.print(table)

    def print_patch_details(self) -> None:
        """Display a detailed table of all generated patches.

        Prints a comprehensive table showing all patch candidates with their
        IDs, generating agents, target files, confidence scores, and current
        status for detailed analysis and comparison.
        """
        patches = self.experiment_data.get("patches", [])

        if not patches:
            self.console.print("[yellow]No patches found in experiment data[/yellow]")
            return

        # Create patches table
        table = Table(title="Generated Patches")
        table.add_column("ID", style="cyan")
        table.add_column("Agent", style="green")
        table.add_column("File", style="magenta")
        table.add_column("Confidence", justify="right", style="yellow")
        table.add_column("Status", style="blue")

        for patch in patches:
            table.add_row(
                patch.get("id", "Unknown")[:8] + "...",
                patch.get("agent_id", "Unknown"),
                str(Path(patch.get("file_path", "")).name),
                f"{patch.get('confidence_score', 0):.2f}",
                patch.get("status", "Unknown"),
            )

        self.console.print(table)

    def save_report(
        self, output_file: str | Path | None = None, format: str = "markdown"
    ) -> Path:
        """Save a comprehensive report to the specified file format.

        Creates and saves either a detailed Markdown report or structured JSON
        report based on the specified format. Automatically generates timestamped
        filenames if no specific output file is provided.

        Args:
            output_file: Optional path for the output file. If None, generates
                a timestamped filename in the experiment directory.
            format: Output format, either 'markdown' or 'json'.

        Returns:
            Path to the created report file.

        Raises:
            ValueError: If an unsupported format is specified.
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if format == "markdown":
                output_file = self.experiment_dir / f"report_{timestamp}.md"
            elif format == "json":
                output_file = self.experiment_dir / f"report_{timestamp}.json"
            else:
                raise ValueError(f"Unsupported format: {format}")
        else:
            output_file = Path(output_file)

        if format == "markdown":
            report_content = self.generate_detailed_report()
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report_content)
        elif format == "json":
            report_data = {
                "summary": self.generate_summary_report(),
                "experiment_data": self.experiment_data,
                "generated_at": datetime.now().isoformat(),
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return output_file

    def export_statistics(self) -> dict[str, Any]:
        """Extract and calculate comprehensive statistics from experiment data.

        Analyzes all experiment data to produce detailed statistics including
        patch generation metrics, evaluation performance, test results, and
        overall experiment outcomes suitable for further analysis or comparison.

        Returns:
            Dictionary containing comprehensive statistics including:
            - experiment_id and basic metadata
            - duration_seconds and success status
            - patch_statistics with counts and confidence metrics
            - evaluation_statistics with comparison results
            - test_statistics with pass rates and timing
        """
        patches = self.experiment_data.get("patches", [])
        evaluations = self.experiment_data.get("evaluations", [])
        test_results = self.experiment_data.get("test_results", [])
        metadata = self.experiment_data.get("metadata", {})

        # Calculate statistics
        patch_stats = {
            "total_patches": len(patches),
            "avg_confidence": sum(p.get("confidence_score", 0) for p in patches)
            / len(patches)
            if patches
            else 0,
            "patches_by_agent": {},
        }

        for patch in patches:
            agent_id = patch.get("agent_id", "Unknown")
            if agent_id not in patch_stats["patches_by_agent"]:
                patch_stats["patches_by_agent"][agent_id] = 0
            patch_stats["patches_by_agent"][agent_id] += 1

        evaluation_stats = {
            "total_evaluations": len(evaluations),
            "avg_evaluation_confidence": sum(
                e.get("confidence", 0) for e in evaluations
            )
            / len(evaluations)
            if evaluations
            else 0,
        }

        test_stats = {
            "total_tests": len(test_results),
            "tests_passed": sum(1 for t in test_results if t.get("passed", False)),
            "avg_test_duration": sum(t.get("duration_seconds", 0) for t in test_results)
            / len(test_results)
            if test_results
            else 0,
        }

        return {
            "experiment_id": metadata.get("experiment_id"),
            "duration_seconds": self.experiment_data.get("duration_seconds", 0),
            "success": self.experiment_data.get("success", False),
            "winning_patch_id": self.experiment_data.get("winning_patch_id"),
            "patch_statistics": patch_stats,
            "evaluation_statistics": evaluation_stats,
            "test_statistics": test_stats,
        }
