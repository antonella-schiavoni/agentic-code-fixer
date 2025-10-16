"""Utility functions for safe pattern handling in OpenCode API calls.

This module provides utilities for properly escaping patterns to ensure they
work correctly with OpenCode's /find API endpoint, avoiding HTTP 400 errors
from improperly encoded search patterns.
"""

import re
import urllib.parse
from typing import Union


def escape_pattern_for_opencode(pattern: str) -> str:
    """Escape a search pattern for safe use with OpenCode's find API.
    
    This function handles both regex special characters and URL encoding
    to ensure patterns work correctly when sent to OpenCode's /find endpoint.
    
    Args:
        pattern: Raw search pattern that may contain regex special characters
        
    Returns:
        Properly escaped and URL-encoded pattern safe for OpenCode API
        
    Examples:
        >>> escape_pattern_for_opencode("test_func(")
        'test_func%5C%28'
        >>> escape_pattern_for_opencode("import.*module")
        'import%5C%2E%5C%2Amodule'
    """
    # First escape regex special characters
    escaped = re.escape(pattern)
    
    # Then URL encode for safe HTTP transmission
    return urllib.parse.quote(escaped, safe='')


def create_function_search_patterns(function_name: str) -> list[str]:
    """Create a list of escaped search patterns for finding function usages.
    
    Args:
        function_name: Name of the function to search for
        
    Returns:
        List of properly escaped patterns for different usage contexts
    """
    raw_patterns = [
        f"{function_name}(",  # Function calls
        f"from .* import.*{function_name}",  # Import statements  
        f"import.*{function_name}",  # Direct imports
    ]
    
    return [escape_pattern_for_opencode(pattern) for pattern in raw_patterns]


def sanitize_function_name_fallback(function_name: str) -> str:
    """Create a sanitized fallback search pattern when main patterns fail.
    
    Args:
        function_name: Original function name
        
    Returns:
        Sanitized version without special characters that commonly cause issues
    """
    # Remove common problematic characters but keep the core identifier
    sanitized = re.sub(r'[^\w_]', '', function_name)
    return escape_pattern_for_opencode(sanitized)