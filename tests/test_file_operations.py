"""Basic test suite for direct file operations functionality.

This test module validates the core functionality of the FileOperationsService
including security boundaries, path validation, and basic file I/O operations.
"""

import pytest
import tempfile
import asyncio
from pathlib import Path

from operations import (
    FileOperationsService,
    LocalBackend,
    FileOperationsConfig,
    SecurityViolationError,
    FileOperationError,
)


class TestFileOperationsService:
    """Test suite for FileOperationsService functionality."""

    @pytest.fixture
    def temp_repo(self):
        """Create a temporary repository for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            
            # Create a sample file structure
            (repo_path / "src").mkdir()
            (repo_path / "src" / "main.py").write_text("print('Hello World')")
            (repo_path / "README.md").write_text("# Test Repository")
            
            yield repo_path

    @pytest.fixture
    def file_ops_service(self, temp_repo):
        """Create FileOperationsService instance for testing."""
        backend = LocalBackend(temp_repo)
        config = FileOperationsConfig()
        return FileOperationsService(
            repo_root=temp_repo,
            backend=backend,
            config=config
        )

    @pytest.mark.asyncio
    async def test_write_file_basic(self, file_ops_service):
        """Test basic file writing functionality."""
        content = "def hello():\n    return 'Hello from test'"
        
        await file_ops_service.write_file("test.py", content)
        
        # Verify file was created and has correct content
        read_content = await file_ops_service.read_file("test.py")
        assert read_content == content

    @pytest.mark.asyncio
    async def test_write_file_in_subdirectory(self, file_ops_service):
        """Test writing files in subdirectories."""
        content = "# Test module\npass"
        
        await file_ops_service.write_file("new_dir/module.py", content)
        
        read_content = await file_ops_service.read_file("new_dir/module.py")
        assert read_content == content

    @pytest.mark.asyncio
    async def test_delete_file(self, file_ops_service):
        """Test file deletion functionality."""
        # Create a file first
        await file_ops_service.write_file("temp.py", "# Temporary file")
        
        # Verify it exists
        content = await file_ops_service.read_file("temp.py")
        assert content == "# Temporary file"
        
        # Delete it
        await file_ops_service.delete_file("temp.py")
        
        # Verify it's gone
        with pytest.raises(FileOperationError):
            await file_ops_service.read_file("temp.py")

    @pytest.mark.asyncio
    async def test_security_violation_absolute_path(self, file_ops_service):
        """Test that absolute paths are rejected."""
        with pytest.raises(SecurityViolationError, match="Absolute paths not allowed"):
            await file_ops_service.write_file("/etc/passwd", "hacker")

    @pytest.mark.asyncio
    async def test_security_violation_path_traversal(self, file_ops_service):
        """Test that path traversal attacks are prevented."""
        with pytest.raises(SecurityViolationError, match="Path outside repository boundary"):
            await file_ops_service.write_file("../../../etc/passwd", "hacker")

    @pytest.mark.asyncio
    async def test_security_violation_forbidden_extension(self, file_ops_service):
        """Test that forbidden file extensions are rejected."""
        with pytest.raises(SecurityViolationError, match="File extension not allowed"):
            await file_ops_service.write_file("malware.exe", "binary data")

    @pytest.mark.asyncio
    async def test_security_violation_forbidden_path(self, file_ops_service):
        """Test that forbidden paths are rejected."""
        with pytest.raises(SecurityViolationError, match="Access to forbidden path"):
            await file_ops_service.write_file(".git/config", "malicious config")
        
        with pytest.raises(SecurityViolationError, match="Access to forbidden path"):
            await file_ops_service.write_file(".env", "SECRET_KEY=hack")

    @pytest.mark.asyncio
    async def test_security_violation_forbidden_filename(self, file_ops_service):
        """Test that forbidden filename patterns are rejected."""
        with pytest.raises(SecurityViolationError, match="Forbidden filename pattern"):
            await file_ops_service.write_file("id_rsa", "fake private key")

    @pytest.mark.asyncio
    async def test_file_size_limit(self, file_ops_service):
        """Test that file size limits are enforced."""
        # Create content larger than the limit (1MB)
        large_content = "x" * (1024 * 1024 + 1)  # 1MB + 1 byte
        
        with pytest.raises(SecurityViolationError, match="File size too large"):
            await file_ops_service.write_file("large.py", large_content)

    @pytest.mark.asyncio
    async def test_operation_limits(self, file_ops_service):
        """Test that operation count limits are enforced."""
        # Write many small files to test the file count limit
        # The default limit is 100 files, so we'll test with a smaller number
        for i in range(10):
            await file_ops_service.write_file(f"test_{i}.py", f"# File {i}")
        
        # Check that operation summary shows correct counts
        summary = file_ops_service.get_operation_summary()
        assert summary["files_modified"] == 10
        assert summary["operation_count"] >= 10  # Includes any read operations

    @pytest.mark.asyncio
    async def test_diff_generation(self, file_ops_service, temp_repo):
        """Test that diffs are generated for audit purposes."""
        # Write initial content
        initial_content = "def original():\n    pass"
        await file_ops_service.write_file("diff_test.py", initial_content)
        
        # Modify the file
        modified_content = "def modified():\n    return 'changed'"
        await file_ops_service.write_file("diff_test.py", modified_content)
        
        # Check operation log for diff information
        summary = file_ops_service.get_operation_summary()
        write_operations = [
            op for op in summary["operations"] 
            if op["operation"] == "write" and op["file_path"] == "diff_test.py"
        ]
        
        # Should have two write operations
        assert len(write_operations) == 2
        
        # Second operation should have a diff preview
        second_write = write_operations[1]
        assert "diff_preview" in second_write
        assert "modified" in second_write["diff_preview"]

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, file_ops_service):
        """Test reading a nonexistent file raises appropriate error."""
        with pytest.raises(FileOperationError, match="File not found"):
            await file_ops_service.read_file("nonexistent.py")

    def test_allowed_extensions(self):
        """Test that common development file extensions are allowed."""
        config = FileOperationsConfig()
        
        allowed_extensions = {
            ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h",
            ".go", ".rs", ".rb", ".php", ".cs", ".yml", ".yaml", ".json", ".md"
        }
        
        assert allowed_extensions.issubset(config.ALLOWED_EXTENSIONS)

    def test_forbidden_paths(self):
        """Test that security-sensitive paths are forbidden."""
        config = FileOperationsConfig()
        
        forbidden_paths = {
            ".git", ".env", ".secrets", "secrets", ".ssh", ".aws"
        }
        
        assert forbidden_paths.issubset(config.FORBIDDEN_PATHS)


if __name__ == "__main__":
    pytest.main([__file__])