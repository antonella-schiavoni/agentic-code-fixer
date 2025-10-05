"""Central coordination system orchestrating the complete automated code fixing pipeline.

This module implements the primary coordinator that manages the entire lifecycle of
automated code patch generation, evaluation, and testing. It integrates all system
components including code indexing, agent orchestration, patch evaluation, testing,
and comprehensive logging to provide a complete end-to-end solution.

The coordinator supports multiple evaluation methodologies, comprehensive error handling,
and detailed experiment tracking for reproducible and analyzable automated code fixing.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from agents import AgentOrchestrator
from core import Config, EvaluationMethod, ExperimentMetadata, PatchStatus, TestResult
from evaluation import EloRanker, PatchEvaluator
from indexing import CodeIndexer
from logging import ExperimentLogger, ReportGenerator
from patching import PatchApplicator, PatchManager

logger = logging.getLogger(__name__)


class AgenticCodeFixer:
    """Central coordinator orchestrating the complete automated code fixing pipeline.

    This class serves as the primary orchestrator for the entire agentic code fixing
    process, managing the integration and execution of all system components. It coordinates
    code indexing, multi-agent patch generation, sophisticated evaluation, testing,
    and comprehensive experiment tracking.

    The coordinator supports configurable evaluation methodologies including AB testing
    and ELO tournaments, provides robust error handling and rollback capabilities, and
    generates detailed experiment reports for analysis and reproducibility.

    Key responsibilities:
    - Orchestrate the complete patch generation and evaluation workflow
    - Manage component initialization and configuration
    - Coordinate multi-phase experiment execution with proper error handling
    - Provide comprehensive logging and progress tracking
    - Generate detailed reports and statistics for analysis

    Attributes:
        config: System configuration containing all parameters and settings.
        experiment_metadata: Metadata tracking experiment configuration and outcomes.
        code_indexer: Vector-based code indexing system for context retrieval.
        agent_orchestrator: Multi-agent coordination system for patch generation.
        patch_evaluator: Claude-powered system for patch comparison and evaluation.
        patch_applicator: System for applying patches and running validation tests.
        elo_ranker: ELO rating system for tournament-style patch evaluation.
        experiment_logger: Comprehensive logging system for progress and results.
        patch_manager: Persistent storage and management system for patch candidates.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the complete agentic code fixing system with all components.

        Sets up all system components, initializes experiment tracking, creates
        necessary directories, and prepares the system for experiment execution.
        Validates configuration and ensures all dependencies are properly configured.

        Args:
            config: Complete system configuration including repository path,
                agent definitions, evaluation parameters, and output settings.
        """
        self.config = config

        # Ensure output directory exists
        config.ensure_output_dir()

        # Initialize components
        self.code_indexer = CodeIndexer(config.vectordb)
        self.agent_orchestrator = AgentOrchestrator(config)
        self.patch_evaluator = PatchEvaluator(
            config.evaluation,
            claude_api_key=config.claude_api_key
        )
        self.patch_applicator = PatchApplicator(config.testing)
        self.elo_ranker = EloRanker(
            k_factor=config.evaluation.elo_k_factor,
            initial_rating=1200.0
        )

        # Initialize experiment metadata
        self.experiment_metadata = ExperimentMetadata(
            repository_path=config.repository_path,
            problem_description=config.problem_description,
            num_agents=len(config.agents),
            num_patches_generated=config.num_patch_candidates,
            evaluation_method=config.evaluation.method,
        )

        # Initialize experiment logger
        self.experiment_logger = ExperimentLogger(
            config.logging,
            self.experiment_metadata.experiment_id
        )

        # Initialize patch manager
        patch_storage_dir = config.get_output_dir() / self.experiment_metadata.experiment_id / "patches"
        self.patch_manager = PatchManager(patch_storage_dir)

        logger.info(f"Initialized Agentic Code Fixer for experiment {self.experiment_metadata.experiment_id}")

    async def run_experiment(self) -> ExperimentMetadata:
        """Execute the complete automated code fixing experiment workflow.

        Orchestrates the entire multi-phase experiment including codebase indexing,
        parallel patch generation, sophisticated evaluation, and comprehensive testing.
        Provides robust error handling, comprehensive logging, and detailed result
        tracking throughout the process.

        The experiment workflow consists of:
        1. Codebase indexing for context retrieval
        2. Multi-agent patch generation with diverse approaches
        3. Comprehensive patch evaluation using configured methodology
        4. Winning patch application and validation testing
        5. Result logging and comprehensive reporting

        Returns:
            ExperimentMetadata containing complete experiment results, timing
            information, success status, and detailed outcome data.

        Raises:
            RuntimeError: If critical phases fail (no patches generated, no winner determined).
            Exception: For other system failures during experiment execution.
        """
        start_time = datetime.now()
        self.experiment_logger.log_experiment_start(self.experiment_metadata)

        try:
            # Phase 1: Index the codebase
            await self._index_codebase()

            # Phase 2: Generate patch candidates
            patches = await self._generate_patches()

            if not patches:
                raise RuntimeError("No patches were generated")

            # Phase 3: Evaluate patches
            winning_patch = await self._evaluate_patches(patches)

            if not winning_patch:
                raise RuntimeError("No winning patch could be determined")

            # Phase 4: Apply and test the winning patch
            test_result = await self._apply_and_test_patch(winning_patch)

            # Update experiment metadata
            end_time = datetime.now()
            self.experiment_metadata.end_time = end_time
            self.experiment_metadata.total_duration_seconds = (end_time - start_time).total_seconds()
            self.experiment_metadata.winning_patch_id = winning_patch.id
            self.experiment_metadata.success = test_result.passed
            self.experiment_metadata.test_results = test_result.model_dump()

            self.experiment_logger.log_experiment_complete(self.experiment_metadata)

            return self.experiment_metadata

        except Exception as e:
            # Handle experiment failure
            end_time = datetime.now()
            self.experiment_metadata.end_time = end_time
            self.experiment_metadata.total_duration_seconds = (end_time - start_time).total_seconds()
            self.experiment_metadata.success = False
            self.experiment_metadata.error_message = str(e)

            self.experiment_logger.log_error(f"Experiment failed: {e}", e)
            self.experiment_logger.log_experiment_complete(self.experiment_metadata)

            raise

    async def _index_codebase(self) -> None:
        """Index the target codebase to enable semantic code search and context retrieval.

        Processes all source files in the repository to create vector embeddings
        for efficient similarity-based retrieval during patch generation. Handles
        file filtering, language detection, and chunk processing for large files.
        """
        self.experiment_logger.log_info("Starting codebase indexing...")

        with self.experiment_logger.create_progress_context("Indexing codebase...") as progress:
            task = progress.add_task("Indexing files...")

            contexts = self.code_indexer.index_repository(
                repo_path=self.config.repository_path,
                exclude_patterns=self.config.exclude_patterns,
                target_files=self.config.target_files,
            )

            progress.update(task, advance=1)

        self.experiment_logger.log_info(f"Indexed {len(contexts)} code contexts")
        logger.info(f"Codebase indexing completed: {len(contexts)} contexts")

    async def _generate_patches(self) -> List:
        """Generate diverse patch candidates using multi-agent orchestration.

        Coordinates parallel execution of multiple specialized AI agents to generate
        diverse patch candidates for the identified problem. Each agent applies
        different approaches and specializations to maximize solution diversity.

        Returns:
            List of PatchCandidate objects containing proposed solutions with
            confidence scores, descriptions, and metadata.
        """
        self.experiment_logger.log_patch_generation_start(len(self.config.agents))

        patches = await self.agent_orchestrator.generate_patches(
            problem_description=self.config.problem_description,
            code_indexer=self.code_indexer,
            target_files=self.config.target_files, #TODO: Remove target_files
        )

        # Store patches in patch manager
        self.patch_manager.add_patches(patches)

        # Log each generated patch
        for patch in patches:
            self.experiment_logger.log_patch_generated(patch)

        self.experiment_logger.log_patch_generation_complete(patches)

        logger.info(f"Generated {len(patches)} patch candidates")
        return patches

    async def _evaluate_patches(self, patches: List) -> Optional:
        """Evaluate patch candidates and determine the optimal solution.

        Uses the configured evaluation methodology (AB testing or ELO tournament)
        to systematically compare patch candidates and identify the best solution.
        Provides detailed reasoning and confidence scores for all comparisons.

        Args:
            patches: List of PatchCandidate objects to evaluate and compare.

        Returns:
            The PatchCandidate determined to be the best solution, or None if
            evaluation fails to identify a clear winner.

        Raises:
            ValueError: If an unsupported evaluation method is configured.
        """
        if len(patches) < 2:
            self.experiment_logger.log_warning("Insufficient patches for evaluation, returning highest confidence patch")
            return max(patches, key=lambda p: p.confidence_score) if patches else None

        method = self.config.evaluation.method
        self.experiment_logger.log_evaluation_start(len(patches), method.value)

        # Get original code for context
        original_code = self._get_original_code(patches[0].file_path)

        if method == EvaluationMethod.AB_TESTING:
            # Use pairwise AB testing
            winning_patch = await self.patch_evaluator.find_best_patch(
                patches=patches,
                problem_description=self.config.problem_description,
                original_code=original_code,
            )

            # Get evaluation results for logging
            evaluation_results = await self.patch_evaluator.evaluate_patches_pairwise(
                patches=patches,
                problem_description=self.config.problem_description,
                original_code=original_code,
            )

            for result in evaluation_results:
                self.experiment_logger.log_evaluation_result(result)

        elif method == EvaluationMethod.ELO_TOURNAMENT:
            # Use ELO tournament
            self.elo_ranker.initialize_patch_ratings(patches)

            # Generate evaluation results for ELO ranking
            #TODO: Does it make sense to use the same evaluation method for ELO ranking?
            evaluation_results = await self.patch_evaluator.evaluate_patches_pairwise(
                patches=patches,
                problem_description=self.config.problem_description,
                original_code=original_code,
            )

            # Update ELO ratings
            self.elo_ranker.update_ratings_from_evaluations(evaluation_results)

            # Get the highest-rated patch
            winning_patch = self.elo_ranker.get_top_patch(patches)

            for result in evaluation_results:
                self.experiment_logger.log_evaluation_result(result)

        else:
            raise ValueError(f"Unsupported evaluation method: {method}")

        # Update patch statuses
        for patch in patches:
            self.patch_manager.update_patch_status(patch.id, PatchStatus.EVALUATED)

        self.experiment_logger.log_evaluation_complete(evaluation_results, winning_patch)

        logger.info(f"Evaluation completed, winning patch: {winning_patch.id}")
        return winning_patch

    async def _apply_and_test_patch(self, patch) -> TestResult:
        """Apply the winning patch and validate it through comprehensive testing.

        Creates an isolated test environment, applies the selected patch, and runs
        the configured test suite to verify that the patch successfully fixes the
        issue without introducing regressions.

        Args:
            patch: PatchCandidate to apply and test.

        Returns:
            TestResult containing test outcomes, timing, and detailed failure
            information if applicable.
        """
        self.experiment_logger.log_test_start(patch.id)

        # Create a test environment
        test_env = self.patch_applicator.create_test_environment(
            repo_path=self.config.repository_path,
            temp_dir=self.config.get_output_dir() / self.experiment_metadata.experiment_id / "test_env"
        )

        try:
            # Apply patch and run tests
            apply_success, test_result = self.patch_applicator.apply_and_test_patch(
                patch=patch,
                repo_path=test_env,
            )

            if apply_success:
                self.patch_manager.update_patch_status(patch.id, PatchStatus.APPLIED)
                if test_result.passed:
                    self.patch_manager.update_patch_status(patch.id, PatchStatus.TESTED)
                else:
                    self.patch_manager.update_patch_status(patch.id, PatchStatus.FAILED)
            else:
                self.patch_manager.update_patch_status(patch.id, PatchStatus.FAILED)

            self.experiment_logger.log_test_result(test_result, patch.id)

            logger.info(f"Patch testing completed: {'PASSED' if test_result.passed else 'FAILED'}")
            return test_result

        finally:
            # Clean up test environment
            self.patch_applicator.cleanup_test_environment(test_env)

    def _get_original_code(self, file_path: str) -> str:
        """Retrieve the original source code from the specified file.

        Reads the current content of the target file to provide context for
        patch evaluation and comparison processes.

        Args:
            file_path: Relative path to the target file within the repository.

        Returns:
            String containing the complete file content, or empty string if
            the file cannot be read.
        """
        try:
            full_path = Path(self.config.repository_path) / file_path
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not read original code from {file_path}: {e}")
            return ""

    def generate_report(self) -> ReportGenerator:
        """Create a comprehensive report generator for experiment analysis.

        Returns:
            ReportGenerator instance configured with experiment data for creating
            detailed reports in multiple formats.
        """
        experiment_dir = self.config.get_output_dir() / self.experiment_metadata.experiment_id
        return ReportGenerator(experiment_dir)

    def get_experiment_statistics(self) -> dict:
        """Collect and compile comprehensive statistics from all system components.

        Aggregates detailed metrics from patch generation, evaluation, testing,
        and orchestration components to provide complete experiment analysis data.

        Returns:
            Dictionary containing comprehensive statistics including experiment
            metadata, patch statistics, orchestrator performance, and evaluation
            results.
        """
        patch_stats = self.patch_manager.get_patch_statistics()
        orchestrator_stats = self.agent_orchestrator.get_orchestrator_stats()

        if self.config.evaluation.method == EvaluationMethod.ELO_TOURNAMENT:
            tournament_stats = self.elo_ranker.get_tournament_stats()
        else:
            tournament_stats = {}

        return {
            "experiment_metadata": self.experiment_metadata.model_dump(),
            "patch_statistics": patch_stats,
            "orchestrator_statistics": orchestrator_stats,
            "tournament_statistics": tournament_stats,
        }

    async def run_baseline_test(self) -> TestResult:
        """Execute baseline tests on the unmodified codebase for comparison.

        Runs the configured test suite on the original codebase to establish
        baseline performance and identify existing test failures for regression
        analysis during patch validation.

        Returns:
            TestResult containing baseline test outcomes for comparison with
            post-patch test results.
        """
        self.experiment_logger.log_info("Running baseline tests...")

        baseline_result = self.patch_applicator.run_tests(
            repo_path=self.config.repository_path
        )

        self.experiment_logger.log_test_result(baseline_result, "baseline")
        return baseline_result

    def export_experiment_data(self, format: str = "json") -> Path:
        """Export comprehensive experiment data to the specified format.

        Creates a complete export of all experiment data including metadata,
        patches, evaluations, and test results for external analysis or archival.

        Args:
            format: Export format (currently supports 'json').

        Returns:
            Path to the created export file.
        """
        return self.experiment_logger.export_logs(format)


async def run_from_config(config_path: str | Path) -> ExperimentMetadata:
    """Convenience function to execute a complete experiment from configuration.

    Loads configuration from the specified file and runs a complete automated
    code fixing experiment, providing a simple interface for experiment execution.

    Args:
        config_path: Path to the YAML or JSON configuration file.

    Returns:
        ExperimentMetadata containing complete experiment results and outcomes.
    """
    from core import load_config

    config = load_config(config_path)
    fixer = AgenticCodeFixer(config) #TODO: Before running the experiment, index the code. Check if the code is indexed before running run_experiment.
    return await fixer.run_experiment()