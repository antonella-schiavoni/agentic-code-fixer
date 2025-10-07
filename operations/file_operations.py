"""Safe file operations service with comprehensive security boundaries.

This module provides a FileOperationsService that enables AI agents to perform
direct file operations while enforcing strict safety constraints. All operations
are sandboxed to prevent unauthorized access outside the repository boundaries.

Key safety features:
- Repository root confinement
- File type and extension validation
- Size limits and operation counting
- Diff generation for audit trails
- Path traversal attack prevention
- Forbidden path protection

The service can operate in multiple modes:
- OpenCode mode: Uses OpenCode SST session-based file operations
- Local mode: Direct filesystem operations for testing/offline use
"""

from __future__ import annotations

import logging
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from core.config import OpenCodeConfig
from opencode_client import OpenCodeClient

logger = logging.getLogger(__name__)


class FileOperationError(Exception):
    """Base exception for file operation failures."""

    pass


class SecurityViolationError(FileOperationError):
    """Raised when a file operation violates security constraints."""

    pass


class FileOperationsConfig:
    """Configuration for file operations safety constraints."""

    # Security constraints
    MAX_FILE_SIZE: int = 1024 * 1024  # 1MB per file
    MAX_TOTAL_SIZE: int = 10 * 1024 * 1024  # 10MB total per session
    MAX_FILE_COUNT: int = 100  # Maximum files that can be modified per session

    # Allowed file extensions
    ALLOWED_EXTENSIONS: set[str] = {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".cs",
        ".swift",
        ".kt",
        ".scala",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".ps1",
        ".yml",
        ".yaml",
        ".json",
        ".toml",
        ".xml",
        ".html",
        ".css",
        ".scss",
        ".sass",
        ".less",
        ".md",
        ".rst",
        ".txt",
        ".cfg",
        ".ini",
        ".conf",
    }

    # Forbidden paths (relative to repo root)
    FORBIDDEN_PATHS: set[str] = {
        ".git",
        ".svn",
        ".hg",
        ".bzr",
        "node_modules",
        "__pycache__",
        ".env",
        ".env.local",
        ".env.production",
        ".env.development",
        ".secret",
        ".secrets",
        "secrets",
        "private",
        ".ssh",
        ".aws",
        ".gcp",
    }

    # Forbidden file patterns
    FORBIDDEN_PATTERNS: set[str] = {
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "*.key",
        "*.pem",
        "*.p12",
        "*.pfx",
        "*.jks",
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "private_key",
    }


class FileOperationsBackend(ABC):
    """Abstract backend for file operations."""

    @abstractmethod
    async def read_file(self, file_path: str) -> str | None:
        """Read file content."""
        pass

    @abstractmethod
    async def write_file(self, file_path: str, content: str) -> bool:
        """Write content to file."""
        pass

    @abstractmethod
    async def delete_file(self, file_path: str) -> bool:
        """Delete a file."""
        pass

    @abstractmethod
    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists."""
        pass


class OpenCodeBackend(FileOperationsBackend):
    """OpenCode SST backend for file operations."""

    def __init__(self, opencode_client: OpenCodeClient, session_id: str) -> None:
        """Initialize OpenCode backend.

        Args:
            opencode_client: OpenCode client instance.
            session_id: OpenCode session ID for operations.
        """
        self.client = opencode_client
        self.session_id = session_id

    async def read_file(self, file_path: str) -> str | None:
        """Read file content via OpenCode."""
        return await self.client.read_file(self.session_id, file_path)

    async def write_file(self, file_path: str, content: str) -> bool:
        """Write content to file via OpenCode."""
        return await self.client.write_file(self.session_id, file_path, content)

    async def delete_file(self, file_path: str) -> bool:
        """Delete file via OpenCode."""
        return await self.client.delete_file(self.session_id, file_path)

    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists via OpenCode."""
        content = await self.read_file(file_path)
        return content is not None


class LocalBackend(FileOperationsBackend):
    """Local filesystem backend for file operations."""

    def __init__(self, repo_root: Path) -> None:
        """Initialize local backend.

        Args:
            repo_root: Root directory for file operations.
        """
        self.repo_root = repo_root

    async def read_file(self, file_path: str) -> str | None:
        """Read file content from local filesystem."""
        try:
            full_path = self.repo_root / file_path
            return full_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return None

    async def write_file(self, file_path: str, content: str) -> bool:
        """Write content to local file."""
        try:
            full_path = self.repo_root / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            return True
        except Exception as e:
            logger.error(f"Failed to write file {file_path}: {e}")
            return False

    async def delete_file(self, file_path: str) -> bool:
        """Delete local file."""
        try:
            full_path = self.repo_root / file_path
            if full_path.exists():
                full_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False

    async def file_exists(self, file_path: str) -> bool:
        """Check if local file exists."""
        full_path = self.repo_root / file_path
        return full_path.exists()


class FileOperationsService:
    """Secure file operations service with comprehensive safety boundaries."""

    def __init__(
        self,
        repo_root: str | Path,
        backend: FileOperationsBackend,
        config: FileOperationsConfig | None = None,
    ) -> None:
        """Initialize file operations service.

        Args:
            repo_root: Repository root directory for security boundary enforcement.
            backend: File operations backend (OpenCode or Local).
            config: Configuration for safety constraints.
        """
        self.repo_root = Path(repo_root).resolve()
        self.backend = backend
        self.config = config or FileOperationsConfig()

        # Session state
        self.files_modified: int = 0
        self.total_bytes_written: int = 0
        self.operation_log: list[dict[str, Any]] = []

        logger.info(f"Initialized FileOperationsService with root: {self.repo_root}")

    def _validate_path(self, file_path: str) -> Path:
        """Validate file path against security constraints.

        Args:
            file_path: Relative path to validate.

        Returns:
            Resolved path if valid.

        Raises:
            SecurityViolationError: If path violates security constraints.
        """
        try:
            path = Path(file_path)

            # Ensure path is relative
            if path.is_absolute():
                raise SecurityViolationError(f"Absolute paths not allowed: {file_path}")

            # Resolve path relative to repo root
            full_path = (self.repo_root / path).resolve()

            # Ensure path stays within repo boundaries
            try:
                full_path.relative_to(self.repo_root)
            except ValueError:
                raise SecurityViolationError(
                    f"Path outside repository boundary: {file_path}"
                )

            # Check for forbidden paths
            for forbidden in self.config.FORBIDDEN_PATHS:
                if str(path).startswith(forbidden) or forbidden in path.parts:
                    raise SecurityViolationError(
                        f"Access to forbidden path: {file_path}"
                    )

            # Check file extension
            if path.suffix and path.suffix.lower() not in self.config.ALLOWED_EXTENSIONS:
                raise SecurityViolationError(
                    f"File extension not allowed: {path.suffix}"
                )

            # Check forbidden patterns
            filename_lower = path.name.lower()
            for pattern in self.config.FORBIDDEN_PATTERNS:
                if pattern.replace("*", "") in filename_lower:
                    raise SecurityViolationError(
                        f"Forbidden filename pattern: {path.name}"
                    )

            return full_path

        except Exception as e:
            if isinstance(e, SecurityViolationError):
                raise
            raise SecurityViolationError(f"Invalid path: {file_path} - {e}")

    def _check_operation_limits(self, content_size: int = 0) -> None:
        """Check if operation would exceed safety limits.

        Args:
            content_size: Size of content to be written.

        Raises:
            SecurityViolationError: If limits would be exceeded.
        """
        if self.files_modified >= self.config.MAX_FILE_COUNT:
            raise SecurityViolationError(
                f"Maximum file count exceeded: {self.config.MAX_FILE_COUNT}"
            )

        if content_size > self.config.MAX_FILE_SIZE:
            raise SecurityViolationError(
                f"File size too large: {content_size} > {self.config.MAX_FILE_SIZE}"
            )

        if self.total_bytes_written + content_size > self.config.MAX_TOTAL_SIZE:
            raise SecurityViolationError(
                f"Total size limit exceeded: {self.config.MAX_TOTAL_SIZE}"
            )

    def _generate_diff(self, file_path: str, old_content: str, new_content: str) -> str:
        """Generate unified diff for audit purposes.

        Args:
            file_path: Path to the file being modified.
            old_content: Original file content.
            new_content: New file content.

        Returns:
            Unified diff string.
        """
        try:
            import difflib

            diff_lines = list(
                difflib.unified_diff(
                    old_content.splitlines(keepends=True),
                    new_content.splitlines(keepends=True),
                    fromfile=f"a/{file_path}",
                    tofile=f"b/{file_path}",
                    n=3,
                )
            )
            return "".join(diff_lines)
        except Exception as e:
            logger.error(f"Failed to generate diff for {file_path}: {e}")
            return f"# Diff generation failed: {e}\n"

    def _log_operation(
        self, operation: str, file_path: str, success: bool, **kwargs: Any
    ) -> None:
        """Log file operation for audit trail.

        Args:
            operation: Type of operation (read, write, delete).
            file_path: Path to the file.
            success: Whether operation succeeded.
            **kwargs: Additional metadata.
        """
        log_entry = {
            "operation": operation,
            "file_path": file_path,
            "success": success,
            "timestamp": logger.handlers[0].formatter.formatTime(
                logging.LogRecord(
                    name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None
                )
            )
            if logger.handlers
            else "unknown",
            **kwargs,
        }
        self.operation_log.append(log_entry)

        if success:
            logger.info(f"File operation: {operation} {file_path}")
        else:
            logger.warning(f"File operation failed: {operation} {file_path}")

    async def read_file(self, file_path: str) -> str:
        """Read file content with security validation.

        Args:
            file_path: Relative path to the file to read.

        Returns:
            File content as string.

        Raises:
            SecurityViolationError: If path violates security constraints.
            FileOperationError: If file cannot be read.
        """
        # Validate path
        validated_path = self._validate_path(file_path)
        relative_path = str(validated_path.relative_to(self.repo_root))

        try:
            content = await self.backend.read_file(relative_path)
            if content is None:
                raise FileOperationError(f"File not found or cannot be read: {file_path}")

            self._log_operation("read", file_path, True, size=len(content))
            return content

        except Exception as e:
            self._log_operation("read", file_path, False, error=str(e))
            if isinstance(e, (SecurityViolationError, FileOperationError)):
                raise
            raise FileOperationError(f"Failed to read file {file_path}: {e}")

    async def write_file(
        self, file_path: str, content: str, *, create_dirs: bool = False
    ) -> None:
        """Write content to file with security validation and audit logging.

        Args:
            file_path: Relative path to the file to write.
            content: Content to write to the file.
            create_dirs: Whether to create parent directories if they don't exist.

        Raises:
            SecurityViolationError: If operation violates security constraints.
            FileOperationError: If file cannot be written.
        """
        # Validate path
        validated_path = self._validate_path(file_path)
        relative_path = str(validated_path.relative_to(self.repo_root))

        # Check operation limits
        content_size = len(content.encode("utf-8"))
        self._check_operation_limits(content_size)

        try:
            # Get current content for diff generation
            old_content = ""
            try:
                old_content = await self.backend.read_file(relative_path) or ""
            except Exception:
                pass  # File might not exist yet

            # Perform the write
            success = await self.backend.write_file(relative_path, content)
            if not success:
                raise FileOperationError(f"Backend failed to write file: {file_path}")

            # Update session state
            self.files_modified += 1
            self.total_bytes_written += content_size

            # Generate diff for audit
            diff = self._generate_diff(file_path, old_content, content)

            self._log_operation(
                "write",
                file_path,
                True,
                size=content_size,
                total_files=self.files_modified,
                total_bytes=self.total_bytes_written,
                diff_preview=diff[:500] + "..." if len(diff) > 500 else diff,
            )

            logger.info(f"Successfully wrote {content_size} bytes to {file_path}")

        except Exception as e:
            self._log_operation("write", file_path, False, error=str(e))
            if isinstance(e, (SecurityViolationError, FileOperationError)):
                raise
            raise FileOperationError(f"Failed to write file {file_path}: {e}")

    async def delete_file(self, file_path: str) -> None:
        """Delete file with security validation.

        Args:
            file_path: Relative path to the file to delete.

        Raises:
            SecurityViolationError: If path violates security constraints.
            FileOperationError: If file cannot be deleted.
        """
        # Validate path
        validated_path = self._validate_path(file_path)
        relative_path = str(validated_path.relative_to(self.repo_root))

        try:
            # Check if file exists and get its content for logging
            file_existed = await self.backend.file_exists(relative_path)
            old_content = ""
            if file_existed:
                try:
                    old_content = await self.backend.read_file(relative_path) or ""
                except Exception:
                    pass

            # Perform the deletion
            success = await self.backend.delete_file(relative_path)
            if not success and file_existed:
                raise FileOperationError(f"Backend failed to delete file: {file_path}")

            # Update session state
            if file_existed:
                self.files_modified += 1

            self._log_operation(
                "delete",
                file_path,
                True,
                existed=file_existed,
                total_files=self.files_modified,
                content_size=len(old_content.encode("utf-8")) if old_content else 0,
            )

            logger.info(f"Successfully deleted {file_path}")

        except Exception as e:
            self._log_operation("delete", file_path, False, error=str(e))
            if isinstance(e, (SecurityViolationError, FileOperationError)):
                raise
            raise FileOperationError(f"Failed to delete file {file_path}: {e}")

    def get_operation_summary(self) -> dict[str, Any]:
        """Get summary of all file operations performed in this session.

        Returns:
            Dictionary containing operation statistics and audit log.
        """
        return {
            "files_modified": self.files_modified,
            "total_bytes_written": self.total_bytes_written,
            "operation_count": len(self.operation_log),
            "operations": self.operation_log,
            "limits": {
                "max_files": self.config.MAX_FILE_COUNT,
                "max_total_size": self.config.MAX_TOTAL_SIZE,
                "max_file_size": self.config.MAX_FILE_SIZE,
            },
        }

    async def validate_code_quality(self, file_path: str) -> bool:
        """Validate code quality of a modified file using linting tools.

        Args:
            file_path: Path to the file to validate.

        Returns:
            True if validation passes, False otherwise.
        """
        try:
            validated_path = self._validate_path(file_path)
            full_path = str(validated_path)

            # Run appropriate linter based on file extension
            if file_path.endswith(".py"):
                # Run ruff for Python files
                result = subprocess.run(
                    ["ruff", "check", full_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    logger.warning(f"Ruff validation failed for {file_path}: {result.stdout}")
                    return False

                # Run pyright for type checking
                result = subprocess.run(
                    ["pyright", full_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    logger.warning(f"Pyright validation failed for {file_path}: {result.stdout}")
                    return False

            # Add more language-specific validations as needed
            return True

        except subprocess.TimeoutExpired:
            logger.error(f"Code quality validation timed out for {file_path}")
            return False
        except Exception as e:
            logger.error(f"Code quality validation failed for {file_path}: {e}")
            return False