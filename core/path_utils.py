"""Path normalization utilities for repository-relative path handling.

This module provides utilities for converting absolute file paths to repository-relative
paths, ensuring consistent path handling across the codebase and preventing issues
with sandbox isolation.
"""

import os
from pathlib import Path


def to_repo_relative(path: str | Path, repo_root: Path) -> str:
    """Convert an absolute or relative path to be relative to the repository root.
    
    This function handles various path formats and ensures that all paths are
    normalized to be relative to the repository root for consistent handling
    in sandbox environments.
    
    Args:
        path: The path to normalize (can be absolute or relative)
        repo_root: The root directory of the repository
        
    Returns:
        A string representing the path relative to repo_root
        
    Examples:
        >>> repo_root = Path("/home/user/project")
        >>> to_repo_relative("/home/user/project/src/main.py", repo_root)
        'src/main.py'
        >>> to_repo_relative("src/main.py", repo_root)
        'src/main.py'
    """
    p = Path(path)
    
    # If the path is already relative, return it as-is
    if not p.is_absolute():
        return str(p)
    
    # For absolute paths, try to make them relative to repo_root
    p_resolved = p.expanduser().resolve()
    repo_root_resolved = repo_root.expanduser().resolve()
    
    try:
        # Try to get relative path if path is within repo_root
        return str(p_resolved.relative_to(repo_root_resolved))
    except ValueError:
        # If path is not within repo_root, use os.path.relpath as fallback
        return os.path.relpath(p_resolved, repo_root_resolved)


def normalize_patch_file_path(patch_file_path: str | Path) -> str:
    """Extract a sensible relative path from an absolute patch file path.
    
    This function attempts to extract meaningful relative paths from absolute
    paths by looking for common project structure patterns. It's used when
    patches contain absolute paths that need to be normalized for sandbox isolation.
    
    Args:
        patch_file_path: The patch file path to normalize
        
    Returns:
        A normalized relative path string
        
    Examples:
        >>> normalize_patch_file_path("/home/user/project/src/main.py")
        'src/main.py'  # if 'src' is recognized as a project pattern
        >>> normalize_patch_file_path("/home/user/project/main.py") 
        'main.py'  # fallback to filename if no patterns found
    """
    patch_file_path = Path(patch_file_path)
    
    if not patch_file_path.is_absolute():
        return str(patch_file_path)
    
    path_parts = patch_file_path.parts
    
    # Look for common project directory patterns to find where the relative path starts
    relative_start_idx = None
    common_patterns = ["src", "lib", "tests", "test", "app", "code"]
    
    # Find the last occurrence of a known project root or take the filename
    for i in range(len(path_parts) - 1, -1, -1):
        part = path_parts[i]
        # If we find a code file, check if parent directories suggest project structure
        if part.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.c', '.h')):
            # For files directly in common patterns, use the pattern + file
            if i > 0 and path_parts[i-1] in common_patterns:
                relative_start_idx = i - 1
                break
            # Otherwise just use the filename (assume it's in repo root)
            elif i == len(path_parts) - 1:  # It's the filename
                relative_start_idx = i
                break
    
    if relative_start_idx is not None:
        return str(Path(*path_parts[relative_start_idx:]))
    else:
        # Fallback: just use the filename
        return patch_file_path.name