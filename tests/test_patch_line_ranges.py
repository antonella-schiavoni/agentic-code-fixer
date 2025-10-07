"""Unit tests for patch applicator line range validation and path normalization.

This module tests the fixes for patch application issues, specifically:
1. Line range validation with proper 0-indexed coordinates
2. Path normalization for sandbox isolation
3. PatchCandidate validation 

These tests ensure that the patch system correctly handles line numbers
as specified in the PatchCandidate documentation (0-indexed) and that
absolute paths are properly converted to relative paths for sandbox safety.
"""

import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.path_utils import to_repo_relative, normalize_patch_file_path
from core.types import PatchCandidate
from core.config import TestingConfig
from patching.patch_applicator import PatchApplicator


class TestPathNormalization:
    """Test path normalization utilities."""
    
    def test_to_repo_relative_with_absolute_path(self):
        """Test converting absolute path within repo to relative path."""
        repo_root = Path("/home/user/project")
        absolute_path = Path("/home/user/project/src/main.py")
        
        result = to_repo_relative(absolute_path, repo_root)
        assert result == "src/main.py"
    
    def test_to_repo_relative_with_relative_path(self):
        """Test that relative paths are returned as-is."""
        repo_root = Path("/home/user/project")
        relative_path = "src/main.py"
        
        result = to_repo_relative(relative_path, repo_root)
        assert result == "src/main.py"
    
    def test_to_repo_relative_outside_repo(self):
        """Test handling of paths outside repository root."""
        repo_root = Path("/home/user/project")
        outside_path = Path("/home/other/file.py")
        
        result = to_repo_relative(outside_path, repo_root)
        # Should return a relative path using os.path.relpath
        assert result == "../../other/file.py"
    
    def test_normalize_patch_file_path_with_src_pattern(self):
        """Test extracting relative path with recognized src pattern."""
        absolute_path = "/home/user/project/src/main.py"
        
        result = normalize_patch_file_path(absolute_path)
        assert result == "src/main.py"
    
    def test_normalize_patch_file_path_fallback_to_filename(self):
        """Test fallback to filename when no patterns recognized."""
        absolute_path = "/some/unknown/structure/file.py"
        
        result = normalize_patch_file_path(absolute_path)
        assert result == "file.py"
    
    def test_normalize_patch_file_path_already_relative(self):
        """Test that relative paths are returned unchanged."""
        relative_path = "src/main.py"
        
        result = normalize_patch_file_path(relative_path)
        assert result == "src/main.py"


class TestPatchCandidateValidation:
    """Test PatchCandidate validation rules."""
    
    def test_patch_candidate_valid_zero_indexed(self):
        """Test creating patch candidate with valid 0-indexed line ranges."""
        patch = PatchCandidate(
            content="def fixed_function():\n    return True",
            description="Fix the function",
            agent_id="test_agent",
            file_path="src/main.py",  # relative path
            line_start=0,  # 0-indexed
            line_end=2,    # 0-indexed, inclusive
            confidence_score=0.9
        )
        
        assert patch.line_start == 0
        assert patch.line_end == 2
        assert patch.file_path == "src/main.py"
    
    def test_patch_candidate_rejects_negative_line_numbers(self):
        """Test that negative line numbers are rejected."""
        with pytest.raises(ValueError, match="Line numbers must be non-negative"):
            PatchCandidate(
                content="code",
                description="test",
                agent_id="test_agent",
                file_path="main.py",
                line_start=-1,  # Invalid
                line_end=2,
                confidence_score=0.9
            )
    
    def test_patch_candidate_rejects_invalid_range(self):
        """Test that line_end < line_start is rejected."""
        with pytest.raises(ValueError, match="line_end \\(1\\) must be >= line_start \\(5\\)"):
            PatchCandidate(
                content="code",
                description="test",
                agent_id="test_agent",
                file_path="main.py",
                line_start=5,
                line_end=1,  # Invalid: less than line_start
                confidence_score=0.9
            )
    
    def test_patch_candidate_rejects_absolute_path(self):
        """Test that absolute file paths are rejected."""
        with pytest.raises(ValueError, match="file_path must be relative to repository root"):
            PatchCandidate(
                content="code",
                description="test",
                agent_id="test_agent",
                file_path="/absolute/path/to/file.py",  # Invalid
                line_start=0,
                line_end=2,
                confidence_score=0.9
            )


class TestPatchApplicatorLineRanges:
    """Test patch application with correct line range handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = TestingConfig(
            test_command="echo 'test passed'",
            test_timeout_seconds=30
        )
        self.applicator = PatchApplicator(self.config)
    
    def test_apply_patch_valid_range(self):
        """Test applying patch within valid line range."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            test_file = repo_path / "test_file.py"
            
            # Create test file with 5 lines (0-4 in 0-indexed)
            original_content = "line 0\nline 1\nline 2\nline 3\nline 4\n"
            test_file.write_text(original_content)
            
            # Create patch that replaces lines 1-2 (0-indexed)
            patch = PatchCandidate(
                content="new line 1\nnew line 2\n",
                description="Replace middle lines",
                agent_id="test_agent",
                file_path="test_file.py",
                line_start=1,  # 0-indexed
                line_end=2,    # 0-indexed, inclusive
                confidence_score=0.9
            )
            
            # Apply patch
            result = self.applicator.apply_patch(patch, repo_path)
            assert result is True
            
            # Verify result
            modified_content = test_file.read_text()
            expected = "line 0\nnew line 1\nnew line 2\nline 3\nline 4\n"
            assert modified_content == expected
    
    def test_apply_patch_out_of_bounds_fails_fast(self):
        """Test that out-of-bounds patches fail fast with meaningful error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            test_file = repo_path / "test_file.py" 
            
            # Create test file with 3 lines (0-2 in 0-indexed)
            test_file.write_text("line 0\nline 1\nline 2\n")
            
            # Create patch that tries to access line 5 (out of bounds)
            patch = PatchCandidate(
                content="new content",
                description="Out of bounds patch",
                agent_id="test_agent",
                file_path="test_file.py",
                line_start=5,  # Out of bounds
                line_end=6,    # Out of bounds
                confidence_score=0.9
            )
            
            # Patch application should fail fast
            result = self.applicator.apply_patch(patch, repo_path)
            assert result is False
            
            # Original file should be unchanged
            content = test_file.read_text()
            assert content == "line 0\nline 1\nline 2\n"
    
    def test_apply_patch_single_line(self):
        """Test applying patch to single line (line_start == line_end)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            test_file = repo_path / "test_file.py"
            
            # Create test file with 3 lines
            test_file.write_text("line 0\nline 1\nline 2\n")
            
            # Create patch that replaces only line 1
            patch = PatchCandidate(
                content="replaced line 1\n",
                description="Replace single line",
                agent_id="test_agent", 
                file_path="test_file.py",
                line_start=1,  # 0-indexed
                line_end=1,    # Same as line_start for single line
                confidence_score=0.9
            )
            
            # Apply patch
            result = self.applicator.apply_patch(patch, repo_path)
            assert result is True
            
            # Verify result
            modified_content = test_file.read_text()
            expected = "line 0\nreplaced line 1\nline 2\n"
            assert modified_content == expected
    
    def test_apply_patch_handles_path_normalization(self):
        """Test that apply_patch method can normalize paths when needed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            test_file = repo_path / "test_file.py"
            test_file.write_text("line 0\n")
            
            # Test the path normalization directly
            from core.path_utils import normalize_patch_file_path
            
            absolute_path = str(test_file.absolute())
            normalized = normalize_patch_file_path(absolute_path)
            
            # Should extract just the filename since no recognizable patterns
            assert normalized == "test_file.py"
            
            # Test with a path that has recognizable patterns
            mock_absolute = "/home/user/project/src/main.py"
            normalized_src = normalize_patch_file_path(mock_absolute)
            assert normalized_src == "src/main.py"
    
    def test_apply_patch_append_new_lines(self):
        """Test appending new lines beyond the current file length."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            test_file = repo_path / "test_file.py"
            
            # Create test file with 3 lines (0-2)
            test_file.write_text("line 0\nline 1\nline 2\n")
            
            # Create patch that appends beyond current file length
            patch = PatchCandidate(
                content="new line 3\nnew line 4\n",
                description="Append new lines",
                agent_id="test_agent",
                file_path="test_file.py",
                line_start=3,  # Beyond current file length (file has lines 0-2)
                line_end=4,    # Appending 2 new lines
                confidence_score=0.9
            )
            
            # Apply patch
            result = self.applicator.apply_patch(patch, repo_path)
            assert result is True
            
            # Verify result
            modified_content = test_file.read_text()
            expected = "line 0\nline 1\nline 2\nnew line 3\nnew line 4\n"
            assert modified_content == expected


if __name__ == "__main__":
    pytest.main([__file__])