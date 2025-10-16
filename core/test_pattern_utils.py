"""Unit tests for pattern utility functions."""

import pytest
from core.pattern_utils import (
    escape_pattern_for_opencode,
    create_function_search_patterns, 
    sanitize_function_name_fallback
)


class TestEscapePatternForOpencode:
    """Test pattern escaping for OpenCode API safety."""
    
    def test_simple_function_name(self):
        """Test escaping a simple function name."""
        result = escape_pattern_for_opencode("test_func")
        assert result == "test_func"
    
    def test_function_with_parenthesis(self):
        """Test escaping function name with parenthesis."""
        result = escape_pattern_for_opencode("test_func(")
        # Should escape the ( and then URL encode the backslash and parenthesis
        assert "%28" in result  # URL encoded (
        assert "%5C" in result  # URL encoded \
    
    def test_regex_special_characters(self):
        """Test escaping common regex special characters."""
        result = escape_pattern_for_opencode("import.*module")
        # Should contain escaped backslashes and the original characters
        assert "%5C" in result  # URL encoded backslash from re.escape()
        assert "module" in result
        
    def test_complex_pattern(self):
        """Test escaping a complex regex pattern."""
        pattern = "from .* import.*test_register_payment"
        result = escape_pattern_for_opencode(pattern)
        
        # Should be properly escaped and URL encoded
        assert "from" in result
        assert "import" in result
        assert "test_register_payment" in result
        # Should contain escaped backslashes from re.escape()
        assert "%5C" in result  # URL encoded backslash
    
    def test_empty_string(self):
        """Test escaping an empty string."""
        result = escape_pattern_for_opencode("")
        assert result == ""


class TestCreateFunctionSearchPatterns:
    """Test function search pattern creation."""
    
    def test_simple_function(self):
        """Test creating patterns for a simple function name."""
        patterns = create_function_search_patterns("my_func")
        
        assert len(patterns) == 3
        # All patterns should be properly escaped
        for pattern in patterns:
            assert "my_func" in pattern
    
    def test_function_with_underscores(self):
        """Test creating patterns for function with underscores."""
        patterns = create_function_search_patterns("test_register_payment")
        
        assert len(patterns) == 3
        for pattern in patterns:
            assert "test_register_payment" in pattern
            
        # First pattern should be for function calls
        assert "%28" in patterns[0]  # URL encoded (
        
        # Other patterns should be for imports
        assert "import" in patterns[1] or "import" in patterns[2]


class TestSanitizeFunctionNameFallback:
    """Test fallback sanitization for problematic function names."""
    
    def test_simple_function_name(self):
        """Test sanitizing a simple function name."""
        result = sanitize_function_name_fallback("my_func")
        assert "my_func" in result
    
    def test_function_with_special_chars(self):
        """Test sanitizing function name with special characters."""
        result = sanitize_function_name_fallback("test_func!()")
        
        # Should remove special characters but keep alphanumeric and underscore
        assert "test_func" in result
        assert "!" not in result
        assert "(" not in result
        assert ")" not in result
    
    def test_already_clean_name(self):
        """Test sanitizing an already clean function name."""
        result = sanitize_function_name_fallback("clean_function_name_123")
        assert "clean_function_name_123" in result