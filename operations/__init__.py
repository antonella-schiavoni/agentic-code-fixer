"""Operations module for direct file manipulation and other operational tasks.

This module provides services for direct file operations with comprehensive
security boundaries, enabling AI agents to perform file I/O operations
safely within repository constraints.
"""

from .file_operations import (
    FileOperationError,
    FileOperationsBackend,
    FileOperationsConfig,
    FileOperationsService,
    LocalBackend,
    OpenCodeBackend,
    SecurityViolationError,
)

__all__ = [
    "FileOperationError",
    "FileOperationsBackend", 
    "FileOperationsConfig",
    "FileOperationsService",
    "LocalBackend",
    "OpenCodeBackend",
    "SecurityViolationError",
]