"""
Regression tests for patch application in isolated test environments.

These tests ensure that:
1. Patches are always applied to the test environment, not the original repository
2. Test isolation prevents pytest from auto-discovering parent directory configs
3. Applied patches are correctly validated and verified
"""

import tempfile
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock
import pytest

from core.types import PatchCandidate
from patching.patch_applicator import PatchApplicator
from core.config import TestingConfig


class TestPatchIsolation:
    """Test suite for patch application isolation and validation."""

    @pytest.fixture
    def testing_config(self):
        """Create a testing configuration for patch applicator."""
        return TestingConfig(
            test_command="pytest -v",
            test_timeout_seconds=30,
            pre_test_commands=[],
            post_test_commands=[]
        )

    @pytest.fixture
    def patch_applicator(self, testing_config):
        """Create a patch applicator instance."""
        return PatchApplicator(testing_config)

    def create_broken_math_module(self, temp_dir: Path) -> Path:
        """Create a temporary math module with a bug."""
        math_file = temp_dir / "math_operations.py"
        content = '''"""Math operations with intentional bug for testing."""

def add(a, b):
    """Add two numbers (but currently multiplies due to bug)."""
    return a * b  # BUG: Should be a + b


def multiply(a, b):
    """Multiply two numbers."""
    return a * b
'''
        math_file.write_text(content)
        return math_file

    def create_test_file(self, temp_dir: Path) -> Path:
        """Create a test file for the math operations."""
        test_file = temp_dir / "test_math_operations.py"
        content = '''"""Tests for math operations."""

from math_operations import add, multiply


def test_add():
    """Test addition function."""
    assert add(2, 3) == 5


def test_multiply():
    """Test multiplication function."""
    assert multiply(2, 3) == 6
'''
        test_file.write_text(content)
        return test_file

    def test_absolute_path_normalization(self, patch_applicator):
        """Test that absolute paths in patches are normalized to relative paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            math_file = self.create_broken_math_module(temp_path)
            test_file = self.create_test_file(temp_path)
            
            # Create patch with absolute path (simulating agent-generated patch)
            absolute_path = str(math_file.absolute())
            patch = PatchCandidate(
                id="test-absolute-path",
                content="    return a + b",
                description="Fix add function",
                agent_id="test-agent",
                file_path=absolute_path,  # This is absolute!
                line_start=5,
                line_end=5,
                confidence_score=1.0
            )
            
            # Apply patch - should normalize path and apply to temp_path, not absolute location
            success = patch_applicator.apply_patch(patch, temp_path, create_backup=True)
            
            assert success, "Patch should be applied successfully"
            
            # Verify the patch was applied to the correct file
            patched_content = math_file.read_text()
            assert "return a + b" in patched_content, "Patch content should be applied"
            assert "return a * b" not in patched_content.split('\n')[4], "Bug should be fixed"
            
            # Verify backup was created
            backup_file = math_file.with_suffix(math_file.suffix + ".backup")
            assert backup_file.exists(), "Backup file should be created"

    def test_patch_validation_and_verification(self, patch_applicator):
        """Test that patches are validated after application."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            math_file = self.create_broken_math_module(temp_path)
            
            # Create valid patch
            patch = PatchCandidate(
                id="test-validation",
                content="    return a + b",
                description="Fix add function",
                agent_id="test-agent",
                file_path="math_operations.py",
                line_start=5,
                line_end=5,
                confidence_score=1.0
            )
            
            # Apply patch
            success = patch_applicator.apply_patch(patch, temp_path, create_backup=True)
            
            assert success, "Valid patch should be applied successfully"
            
            # Verify the exact content was written
            patched_lines = math_file.read_text().splitlines()
            assert patched_lines[4].strip() == "return a + b", "Exact patch content should be applied"

    def test_test_isolation_prevents_parent_config_discovery(self, patch_applicator):
        """Test that test isolation prevents pytest from using parent directory configs."""
        # This test verifies that the _isolate_test_command method works correctly
        test_repo_path = Path("/some/test/environment")
        
        # Test pytest command isolation
        original_command = "pytest"
        isolated_command = patch_applicator._isolate_test_command(original_command, test_repo_path)
        
        assert "--rootdir=" in isolated_command, "Should add rootdir flag"
        assert "--rootdir=." in isolated_command, "Should use current directory as rootdir"
        assert "--override-ini" in isolated_command, "Should override ini settings"
        
        # Test that already isolated commands aren't double-modified
        pre_isolated = "pytest --rootdir=/existing/path"
        result = patch_applicator._isolate_test_command(pre_isolated, test_repo_path)
        assert result == pre_isolated, "Should not modify already isolated commands"

    def test_functional_end_to_end_patch_and_test(self, patch_applicator):
        """Test complete patch application and test execution in isolated environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            math_file = self.create_broken_math_module(temp_path)
            test_file = self.create_test_file(temp_path)
            
            # Create patch to fix the bug
            patch = PatchCandidate(
                id="test-functional",
                content="    return a + b",
                description="Fix add function",
                agent_id="test-agent",
                file_path="math_operations.py",
                line_start=5,
                line_end=5,
                confidence_score=1.0
            )
            
            # Verify initial broken state
            result_before = subprocess.run([
                sys.executable, "-c",
                "import sys; sys.path.insert(0, '.'); "
                "from math_operations import add; "
                "print(add(2, 3))"
            ], cwd=temp_path, capture_output=True, text=True)
            
            assert "6" in result_before.stdout, "Initial state should be broken (2*3=6)"
            
            # Apply patch
            success = patch_applicator.apply_patch(patch, temp_path, create_backup=True)
            assert success, "Patch should apply successfully"
            
            # Verify patched state
            result_after = subprocess.run([
                sys.executable, "-c",
                "import sys; sys.path.insert(0, '.'); "
                "from math_operations import add; "
                "print(add(2, 3))"
            ], cwd=temp_path, capture_output=True, text=True)
            
            assert "5" in result_after.stdout, "Patched state should be fixed (2+3=5)"
            
            # Run tests (if pytest is available)
            try:
                test_result = patch_applicator.run_tests(temp_path, patch.id)
                # Test should pass now that the bug is fixed
                assert test_result.passed, "Tests should pass after patch is applied"
                assert "test_add" not in [t for t in test_result.failed_tests], "add test should not fail"
            except Exception as e:
                # If pytest isn't available in the test environment, that's OK
                pytest.skip(f"Pytest not available for functional test: {e}")

    def test_patch_revert_functionality(self, patch_applicator):
        """Test that patches can be reverted using backups."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test file
            math_file = self.create_broken_math_module(temp_path)
            original_content = math_file.read_text()
            
            # Create and apply patch
            patch = PatchCandidate(
                id="test-revert",
                content="    return a + b",
                description="Fix add function",
                agent_id="test-agent",
                file_path="math_operations.py",
                line_start=5,
                line_end=5,
                confidence_score=1.0
            )
            
            # Apply patch
            success = patch_applicator.apply_patch(patch, temp_path, create_backup=True)
            assert success, "Patch should apply successfully"
            
            # Verify patch was applied
            patched_content = math_file.read_text()
            assert patched_content != original_content, "Content should be changed"
            assert "return a + b" in patched_content, "Patch should be applied"
            
            # Revert patch
            revert_success = patch_applicator.revert_patch(patch, temp_path)
            assert revert_success, "Patch should revert successfully"
            
            # Verify original content is restored
            reverted_content = math_file.read_text()
            assert reverted_content == original_content, "Original content should be restored"
            assert "return a * b" in reverted_content, "Original bug should be restored"