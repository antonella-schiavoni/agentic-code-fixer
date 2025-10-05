"""Advanced patch application system with integrated testing and validation capabilities.

This module provides comprehensive patch application functionality that safely applies
code changes to target repositories while maintaining data integrity through backup
systems. It includes robust testing integration to verify that patches successfully
fix issues without introducing regressions.

The applicator supports multiple testing frameworks, provides detailed test result
analysis, and offers temporary test environment management for safe patch validation.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from core.config import OpenCodeConfig, TestingConfig
from core.types import PatchCandidate, TestResult
from opencode_client import OpenCodeClient

logger = logging.getLogger(__name__)


class PatchApplicator:
    """Advanced patch application system with integrated testing and validation.

    This class provides comprehensive functionality for safely applying code patches
    to target repositories while maintaining data integrity and verification standards.
    It supports backup creation, rollback capabilities, and integrated testing to
    ensure patches successfully address issues without introducing regressions.

    The applicator can leverage OpenCode SST for shell execution when available,
    providing better isolation and execution tracking. It falls back to local
    subprocess execution when OpenCode is not available or disabled.

    Attributes:
        config: Testing configuration including commands, timeouts, and validation rules.
        opencode_config: Optional OpenCode configuration for shell execution.
        opencode_client: Optional OpenCode client for session-based testing.
    """

    def __init__(
        self,
        config: TestingConfig,
        opencode_config: OpenCodeConfig | None = None
    ) -> None:
        """Initialize the patch applicator with testing and validation configuration.

        Sets up the applicator with specified testing parameters, timeout settings,
        and validation rules for patch application and verification processes.
        Optionally configures OpenCode integration for enhanced shell execution.

        Args:
            config: TestingConfig object containing test commands, timeout settings,
                and regression detection parameters.
            opencode_config: Optional OpenCode configuration for shell execution.
        """
        self.config = config
        self.opencode_config = opencode_config
        self.opencode_client = None

        if opencode_config and opencode_config.enable_shell_execution:
            self.opencode_client = OpenCodeClient(opencode_config)

        logger.info("Initialized patch applicator with OpenCode integration: %s",
                   bool(self.opencode_client))

    def apply_patch(
        self,
        patch: PatchCandidate,
        repo_path: str | Path,
        create_backup: bool = True,
    ) -> bool:
        """Apply a code patch to the target repository with optional backup creation.

        Safely applies the specified patch to the target file by replacing the
        designated line range with the patch content. Creates automatic backups
        and provides rollback capability in case of application failures.

        Args:
            patch: PatchCandidate containing the code changes and target location.
            repo_path: Path to the repository root directory.
            create_backup: Whether to create a backup file before applying changes.

        Returns:
            True if the patch was successfully applied, False otherwise.
        """
        repo_path = Path(repo_path)
        target_file = repo_path / patch.file_path

        if not target_file.exists():
            logger.error(f"Target file does not exist: {target_file}")
            return False

        try:
            # Create backup if requested
            backup_file = None
            if create_backup:
                backup_file = target_file.with_suffix(target_file.suffix + ".backup")
                shutil.copy2(target_file, backup_file)
                logger.info(f"Created backup: {backup_file}")

            # Read current file content
            with open(target_file, encoding="utf-8") as f:
                lines = f.readlines()

            # Apply patch by replacing lines
            if patch.line_end >= len(lines):
                logger.error(f"Patch line range exceeds file length: {patch.line_end} >= {len(lines)}")
                return False

            # Replace the specified line range with patch content
            patch_lines = patch.content.split("\n")
            if not patch.content.endswith("\n"):
                patch_lines = [line + "\n" for line in patch_lines[:-1]] + [patch_lines[-1]]
            else:
                patch_lines = [line + "\n" for line in patch_lines]

            # Apply the patch
            new_lines = (
                lines[:patch.line_start] +
                patch_lines +
                lines[patch.line_end + 1:]
            )

            # Write modified content back to file
            with open(target_file, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

            logger.info(f"Applied patch {patch.id} to {target_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to apply patch {patch.id}: {e}")

            # Restore backup if it exists
            if backup_file and backup_file.exists():
                shutil.copy2(backup_file, target_file)
                logger.info("Restored file from backup")

            return False

    def run_tests(
        self,
        repo_path: str | Path,
        patch_id: str | None = None,
    ) -> TestResult:
        """Execute the configured test suite and return detailed results.

        Runs the complete test pipeline including pre-test setup, main test execution,
        and post-test cleanup. Captures comprehensive output and timing information
        for analysis and reporting purposes.

        Args:
            repo_path: Path to the repository where tests should be executed.
            patch_id: Optional identifier of the applied patch for tracking purposes.

        Returns:
            TestResult object containing exit codes, output, timing, and parsed
            failure information.
        """
        repo_path = Path(repo_path)
        start_time = datetime.now()

        try:
            # Run pre-test commands
            for cmd in self.config.pre_test_commands:
                self._run_command(cmd, repo_path)

            # Run main test command
            #TODO: We might need an LLM call with MCP to understand how to run the tests.
            result = self._run_command(
                self.config.test_command,
                repo_path,
                timeout=self.config.test_timeout_seconds
            )

            # Run post-test commands
            for cmd in self.config.post_test_commands:
                self._run_command(cmd, repo_path)

            # Parse test results
            passed = result.returncode == 0
            failed_tests = self._parse_failed_tests(result.stderr + result.stdout)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            test_result = TestResult(
                test_command=self.config.test_command,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_seconds=duration,
                passed=passed,
                failed_tests=failed_tests,
            )

            logger.info(
                f"Test run completed: {'PASSED' if passed else 'FAILED'} "
                f"(duration: {duration:.2f}s)"
            )

            return test_result

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.error(f"Test run failed with exception: {e}")

            return TestResult(
                test_command=self.config.test_command,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
                passed=False,
                failed_tests=[],
            )

    def apply_and_test_patch(
        self,
        patch: PatchCandidate,
        repo_path: str | Path,
        baseline_test_result: TestResult | None = None,
    ) -> tuple[bool, TestResult]:
        """Apply a patch and immediately run tests to validate the changes.

        Combines patch application and testing into a single atomic operation,
        providing comprehensive validation of the patch effectiveness. Includes
        regression detection when baseline test results are provided.

        Args:
            patch: PatchCandidate to apply and test.
            repo_path: Path to the target repository.
            baseline_test_result: Optional baseline test results for regression detection.

        Returns:
            Tuple containing:
            - bool: Whether the patch was successfully applied
            - TestResult: Comprehensive test execution results
        """
        repo_path = Path(repo_path)

        # Apply the patch
        apply_success = self.apply_patch(patch, repo_path, create_backup=True)
        if not apply_success:
            # Return failed test result if patch couldn't be applied
            return False, TestResult(
                test_command=self.config.test_command,
                exit_code=-1,
                stdout="",
                stderr="Failed to apply patch",
                duration_seconds=0.0,
                passed=False,
                failed_tests=[],
            )

        # Run tests
        test_result = self.run_tests(repo_path, patch.id)

        # Check for regressions if baseline is provided
        if baseline_test_result and self.config.fail_on_regression:
            test_result.new_failures = self._find_new_failures(
                baseline_test_result.failed_tests,
                test_result.failed_tests
            )

        return apply_success, test_result

    def create_test_environment(
        self,
        repo_path: str | Path,
        temp_dir: str | Path | None = None,
    ) -> Path:
        """Create an isolated test environment for safe patch validation.

        Creates a complete copy of the repository in a temporary location, allowing
        for safe patch testing without affecting the original codebase. The isolated
        environment includes all source files but excludes version control data.

        Args:
            repo_path: Path to the source repository to copy.
            temp_dir: Optional directory for the test environment. If not specified,
                uses system temporary directory.

        Returns:
            Path to the created test environment directory.
        """
        #TODO: This can be improved. Think about large repositories, monorepos, etc. Does it make sense to copy the entire repository? If it's a git repository, we might be able to use worktrees.
        #TODO: What happens with the dependencies?
        repo_path = Path(repo_path)

        if temp_dir:
            temp_path = Path(temp_dir)
            temp_path.mkdir(parents=True, exist_ok=True)
            test_env = temp_path / f"test_env_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            test_env = Path(tempfile.mkdtemp(prefix="agentic_code_fixer_"))

        # Copy repository to test environment
        shutil.copytree(repo_path, test_env, ignore=shutil.ignore_patterns(".git"))

        logger.info(f"Created test environment: {test_env}")
        return test_env

    def cleanup_test_environment(self, test_env_path: str | Path) -> None:
        """Remove a temporary test environment and all its contents.

        Safely deletes the specified test environment directory and all contained
        files, freeing up disk space after test completion.

        Args:
            test_env_path: Path to the test environment directory to remove.
        """
        test_env_path = Path(test_env_path)
        if test_env_path.exists():
            shutil.rmtree(test_env_path)
            logger.info(f"Cleaned up test environment: {test_env_path}")

    def revert_patch(
        self,
        patch: PatchCandidate,
        repo_path: str | Path,
    ) -> bool:
        """Revert a previously applied patch using the automatic backup.

        Restores the target file to its state before patch application using
        the automatically created backup file. This provides a clean rollback
        mechanism when patches need to be undone.

        Args:
            patch: PatchCandidate that was previously applied.
            repo_path: Path to the repository containing the modified file.

        Returns:
            True if the patch was successfully reverted, False otherwise.
        """
        repo_path = Path(repo_path)
        target_file = repo_path / patch.file_path
        backup_file = target_file.with_suffix(target_file.suffix + ".backup")

        if not backup_file.exists():
            logger.error(f"No backup found for {target_file}")
            return False

        try:
            shutil.copy2(backup_file, target_file)
            backup_file.unlink()  # Remove backup after successful revert
            logger.info(f"Reverted patch {patch.id} for {target_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to revert patch {patch.id}: {e}")
            return False

    def _run_command(
        self,
        command: str,
        cwd: Path,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess:
        """Execute a shell command with proper error handling and timeout control.

        Runs the specified command in the given working directory, capturing
        both stdout and stderr output. Provides timeout control to prevent
        hanging processes and comprehensive error handling.

        Args:
            command: Shell command string to execute.
            cwd: Working directory for command execution.
            timeout: Optional timeout in seconds for command execution.

        Returns:
            CompletedProcess object containing exit code and captured output.

        Raises:
            subprocess.TimeoutExpired: If command execution exceeds timeout.
            Exception: If command execution fails for other reasons.
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            logger.debug(f"Command '{command}' completed with exit code {result.returncode}")
            return result

        except subprocess.TimeoutExpired:
            logger.error(f"Command '{command}' timed out after {timeout} seconds")
            raise

        except Exception as e:
            logger.error(f"Failed to run command '{command}': {e}")
            raise

    def _parse_failed_tests(self, output: str) -> list[str]:
        """Extract failed test names from test framework output using pattern matching.

        Analyzes test output using regular expressions to identify failed tests
        across multiple testing frameworks including pytest, unittest, Jest, and
        Go test. Provides normalized test failure information for analysis.

        Args:
            output: Raw test output containing failure information.

        Returns:
            List of failed test names extracted from the output.
        """
        failed_tests = []

        # Common patterns for different test frameworks
        patterns = [
            # pytest
            r"FAILED (.+?) -",
            r"(.+?)::.*FAILED",
            # unittest
            r"FAIL: (.+?) \(",
            # Jest
            r"✕ (.+)",
            # Go test
            r"--- FAIL: (.+?) \(",
        ]

        import re
        for pattern in patterns:
            matches = re.findall(pattern, output)
            failed_tests.extend(matches)

        # Remove duplicates and clean up
        failed_tests = list(set(failed_tests))
        failed_tests = [test.strip() for test in failed_tests if test.strip()]

        return failed_tests

    def _find_new_failures(
        self,
        baseline_failures: list[str],
        current_failures: list[str]
    ) -> list[str]:
        """Identify regression failures by comparing current results with baseline.

        Compares the current test failures against a known baseline to identify
        new failures that may have been introduced by the applied patch. This
        helps detect regressions and validate patch quality.

        Args:
            baseline_failures: List of test names that failed in the baseline run.
            current_failures: List of test names that failed in the current run.

        Returns:
            List of test names that failed in current run but not in baseline,
            indicating potential regressions.
        """
        baseline_set = set(baseline_failures)
        current_set = set(current_failures)
        return list(current_set - baseline_set)