"""Unit tests for the TestSelector class.

These tests verify that the TestSelector correctly:
1. Extracts modified symbols from patch candidates
2. Builds and maintains test symbol indexes 
3. Selects relevant tests based on patch analysis
4. Handles edge cases and error conditions gracefully
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from core.types import PatchCandidate
from testing.test_selector import TestSelector


class TestTestSelector:
    """Test suite for the TestSelector class."""

    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create a temporary repository with sample code and tests."""
        # Create main code files
        (tmp_path / "math_operations.py").write_text('''
"""Math operations module."""

def add(a, b):
    """Add two numbers."""
    return a + b

def multiply(a, b):
    """Multiply two numbers."""
    return a * b

class Calculator:
    """Simple calculator class."""
    
    def divide(self, a, b):
        """Divide two numbers."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
''')

        (tmp_path / "string_utils.py").write_text('''
"""String utility functions."""

def reverse_string(s):
    """Reverse a string."""
    return s[::-1]

def capitalize_words(s):
    """Capitalize each word in a string."""
    return ' '.join(word.capitalize() for word in s.split())
''')

        # Create test files that import different functions
        (tmp_path / "test_math.py").write_text('''
"""Tests for math operations."""
from math_operations import add, multiply, Calculator

def test_add():
    assert add(2, 3) == 5

def test_multiply():
    assert multiply(2, 3) == 6

def test_calculator_divide():
    calc = Calculator()
    assert calc.divide(10, 2) == 5
''')

        (tmp_path / "test_strings.py").write_text('''
"""Tests for string utilities."""
from string_utils import reverse_string, capitalize_words

def test_reverse():
    assert reverse_string("hello") == "olleh"

def test_capitalize():
    assert capitalize_words("hello world") == "Hello World"
''')

        (tmp_path / "test_mixed.py").write_text('''
"""Tests that use multiple modules."""
import math_operations
from string_utils import reverse_string

def test_mixed_usage():
    result = math_operations.add(1, 2)
    reversed_str = reverse_string("test")
    assert result == 3
    assert reversed_str == "tset"
''')

        return tmp_path

    @pytest.fixture
    def test_selector(self, temp_repo):
        """Create a TestSelector instance for the temporary repository."""
        return TestSelector(temp_repo)

    def test_init(self, temp_repo):
        """Test TestSelector initialization."""
        selector = TestSelector(
            repository_path=temp_repo,
            cache_dir=temp_repo / ".custom_cache",
            test_patterns=["test_*.py"]
        )
        
        assert selector.repository_path == temp_repo
        assert selector.cache_dir == temp_repo / ".custom_cache"
        assert "test_*.py" in selector.test_patterns
        assert selector._symbol_index is None

    def test_discover_test_files(self, test_selector):
        """Test discovery of test files."""
        test_files = test_selector._discover_test_files()
        test_file_names = [f.name for f in test_files]
        
        assert len(test_files) == 3
        assert "test_math.py" in test_file_names
        assert "test_strings.py" in test_file_names
        assert "test_mixed.py" in test_file_names

    def test_extract_modified_symbols_function_patch(self, test_selector):
        """Test extracting symbols from a patch that modifies a function."""
        patch = PatchCandidate(
            id="test-patch",
            content="def add(a, b):\n    return a + b + 1",  # Modified function
            description="Fix add function", 
            agent_id="test-agent",
            file_path="math_operations.py",
            line_start=3,  # Line containing the add function
            line_end=5,
            confidence_score=1.0
        )
        
        symbols = test_selector._extract_modified_symbols([patch])
        
        assert len(symbols) >= 1
        assert any("add" in symbol for symbol in symbols)
        # Should extract "math_operations.add"
        assert "math_operations.add" in symbols

    def test_extract_modified_symbols_class_patch(self, test_selector):
        """Test extracting symbols from a patch that modifies a class method."""
        patch = PatchCandidate(
            id="test-patch",
            content="    def divide(self, a, b):\n        return a / b if b != 0 else float('inf')",
            description="Fix divide method",
            agent_id="test-agent", 
            file_path="math_operations.py",
            line_start=12,  # Line containing the divide method
            line_end=15,
            confidence_score=1.0
        )
        
        symbols = test_selector._extract_modified_symbols([patch])
        
        assert len(symbols) >= 1
        # Should extract method within Calculator class
        assert any("Calculator" in symbol and "divide" in symbol for symbol in symbols)

    def test_file_path_to_module_name(self, test_selector):
        """Test conversion of file paths to module names."""
        # Simple case
        assert test_selector._file_path_to_module_name(Path("math_operations.py")) == "math_operations"
        
        # Nested path
        assert test_selector._file_path_to_module_name(Path("utils/string_utils.py")) == "utils.string_utils"
        
        # With src prefix (should be removed)
        assert test_selector._file_path_to_module_name(Path("src/core/config.py")) == "core.config"

    def test_parse_symbols_from_content(self, test_selector):
        """Test parsing symbols from code content."""
        content = '''
def test_function():
    pass

class TestClass:
    def test_method(self):
        pass
'''
        
        symbols = test_selector._parse_symbols_from_content(content, "test_module")
        
        assert "test_module.test_function" in symbols
        assert "test_module.TestClass" in symbols
        assert "test_module.TestClass.test_method" in symbols

    def test_parse_symbols_with_syntax_error_fallback(self, test_selector):
        """Test that regex fallback works when AST parsing fails."""
        # Incomplete/invalid Python code that would cause SyntaxError
        content = '''
def incomplete_function(
    # missing closing parenthesis and body

class IncompleteClass:
'''
        
        symbols = test_selector._parse_symbols_from_content(content, "test_module")
        
        # Should fall back to regex and still extract what it can
        assert any("incomplete_function" in symbol for symbol in symbols)
        assert any("IncompleteClass" in symbol for symbol in symbols)

    def test_analyze_test_file_imports(self, test_selector, temp_repo):
        """Test analysis of imports in test files."""
        test_file = temp_repo / "test_math.py"
        
        imports = test_selector._analyze_test_file_imports(test_file)
        
        # Should detect imported functions
        assert "math_operations.add" in imports
        assert "math_operations.multiply" in imports  
        assert "math_operations.Calculator" in imports
        assert "add" in imports  # Unqualified names too
        assert "multiply" in imports
        assert "Calculator" in imports

    def test_build_test_symbol_index(self, test_selector):
        """Test building the complete test symbol index."""
        index = test_selector._build_test_symbol_index()
        
        # Should map symbols to test files that import them
        assert "math_operations.add" in index
        assert "test_math.py" in index["math_operations.add"]
        assert "test_mixed.py" in index["math_operations.add"]  # Also imported in mixed test
        
        assert "string_utils.reverse_string" in index
        assert "test_strings.py" in index["string_utils.reverse_string"]
        assert "test_mixed.py" in index["string_utils.reverse_string"]

    def test_symbols_related(self, test_selector):
        """Test detection of related symbols."""
        # Class and method relationship
        assert test_selector._symbols_related("math_operations.Calculator", "math_operations.Calculator.divide")
        assert test_selector._symbols_related("math_operations.Calculator.divide", "math_operations.Calculator")
        
        # Unrelated symbols
        assert not test_selector._symbols_related("math_operations.add", "string_utils.reverse_string")

    def test_select_tests_for_patches(self, test_selector):
        """Test complete test selection workflow."""
        # Create a patch that modifies the add function
        patch = PatchCandidate(
            id="add-patch",
            content="def add(a, b):\n    return a + b + 1",
            description="Modify add function",
            agent_id="test-agent",
            file_path="math_operations.py", 
            line_start=3,
            line_end=5,
            confidence_score=1.0
        )
        
        selected_tests = test_selector.select_tests_for_patches([patch])
        
        # Should select tests that import/use the add function
        assert len(selected_tests) >= 1
        assert "test_math.py" in selected_tests  # Directly imports add
        assert "test_mixed.py" in selected_tests  # Also uses math_operations.add
        
        # Should not select string-only tests
        assert "test_strings.py" not in selected_tests

    def test_select_tests_for_class_method_patch(self, test_selector):
        """Test test selection for class method patches."""
        # Create a patch that modifies Calculator.divide method
        patch = PatchCandidate(
            id="divide-patch",
            content="    def divide(self, a, b):\n        return a / b if b != 0 else 0",
            description="Fix divide method",
            agent_id="test-agent",
            file_path="math_operations.py",
            line_start=12,
            line_end=15,
            confidence_score=1.0
        )
        
        selected_tests = test_selector.select_tests_for_patches([patch])
        
        # Should select tests that use Calculator
        assert "test_math.py" in selected_tests
        # Should not include tests that don't use Calculator
        assert "test_strings.py" not in selected_tests

    def test_select_tests_no_matches(self, test_selector):
        """Test behavior when no relevant tests are found."""
        # Create a patch for a non-existent file/function
        patch = PatchCandidate(
            id="unknown-patch",
            content="def unknown_function():\n    pass",
            description="Add unknown function",
            agent_id="test-agent",
            file_path="unknown_module.py",
            line_start=1,
            line_end=3,
            confidence_score=1.0
        )
        
        selected_tests = test_selector.select_tests_for_patches([patch])
        
        # Should return empty list (fallback to all tests)
        assert selected_tests == []

    def test_create_test_manifest(self, test_selector, temp_repo):
        """Test creation of test manifest files."""
        test_files = ["test_math.py", "test_mixed.py"]
        
        manifest_path = test_selector.create_test_manifest(test_files)
        
        assert manifest_path.exists()
        content = manifest_path.read_text().strip().split('\n')
        assert "test_math.py" in content
        assert "test_mixed.py" in content
        assert len(content) == 2

    def test_cleanup_manifest(self, test_selector, temp_repo):
        """Test cleanup of test manifest files."""
        test_files = ["test_math.py"]
        manifest_path = test_selector.create_test_manifest(test_files)
        
        assert manifest_path.exists()
        
        test_selector.cleanup_manifest(manifest_path)
        
        assert not manifest_path.exists()

    def test_cache_functionality(self, test_selector, temp_repo):
        """Test that caching works correctly."""
        # Build index first time
        index1 = test_selector._get_test_symbol_index()
        
        # Should save to cache
        assert test_selector.index_cache_file.exists()
        assert test_selector.index_meta_file.exists()
        
        # Create new selector instance
        test_selector2 = TestSelector(temp_repo)
        
        # Should load from cache
        index2 = test_selector2._get_test_symbol_index()
        
        assert index1 == index2

    def test_cache_invalidation_on_file_change(self, test_selector, temp_repo):
        """Test that cache is invalidated when test files change.""" 
        # Build index first
        test_selector._get_test_symbol_index()
        assert test_selector.index_cache_file.exists()
        
        # Modify a test file
        test_file = temp_repo / "test_math.py"
        original_content = test_file.read_text()
        test_file.write_text(original_content + "\n# Modified")
        
        # Cache should be considered stale
        assert not test_selector._is_index_fresh()

    def test_multiple_patches_same_file(self, test_selector):
        """Test handling multiple patches to the same file."""
        patches = [
            PatchCandidate(
                id="patch1",
                content="def add(a, b):\n    return a + b + 1",
                description="Modify add",
                agent_id="agent1",
                file_path="math_operations.py",
                line_start=3,
                line_end=5,
                confidence_score=1.0
            ),
            PatchCandidate(
                id="patch2", 
                content="def multiply(a, b):\n    return a * b * 2",
                description="Modify multiply",
                agent_id="agent2", 
                file_path="math_operations.py",
                line_start=7,
                line_end=9,
                confidence_score=1.0
            )
        ]
        
        selected_tests = test_selector.select_tests_for_patches(patches)
        
        # Should select tests that use either function
        assert "test_math.py" in selected_tests
        assert "test_mixed.py" in selected_tests

    @patch('testing.test_selector.logger')
    def test_error_handling_graceful_fallback(self, mock_logger, test_selector):
        """Test that errors are handled gracefully with fallback behavior."""
        # Create patch with invalid content that might cause issues
        patch = PatchCandidate(
            id="problematic-patch",
            content="",  # Empty content
            description="Empty patch",
            agent_id="test-agent",
            file_path="nonexistent.py", 
            line_start=0,
            line_end=0,
            confidence_score=1.0
        )
        
        # Should not crash and should return empty list (fallback)
        selected_tests = test_selector.select_tests_for_patches([patch])
        assert selected_tests == []
        
        # Should log appropriate debug messages
        assert mock_logger.debug.called or mock_logger.info.called

    def test_generate_submodule_names(self, test_selector):
        """Test generation of submodule names from imports."""
        submodules = test_selector._generate_submodule_names("pkg.subpkg.module")
        
        expected = {"pkg", "pkg.subpkg", "pkg.subpkg.module"}
        assert submodules == expected

    def test_ranges_overlap(self, test_selector):
        """Test line range overlap detection."""
        # Overlapping ranges
        assert test_selector._ranges_overlap(1, 5, 3, 7)
        assert test_selector._ranges_overlap(3, 7, 1, 5)
        
        # Non-overlapping ranges
        assert not test_selector._ranges_overlap(1, 3, 5, 7)
        assert not test_selector._ranges_overlap(5, 7, 1, 3)
        
        # Adjacent ranges (should overlap at boundary)
        assert test_selector._ranges_overlap(1, 3, 3, 5)

    def test_extract_symbols_in_line_range(self, test_selector, temp_repo):
        """Test extraction of symbols that overlap with line ranges."""
        math_file = temp_repo / "math_operations.py"
        
        # Test range that overlaps with the add function
        symbols = test_selector._extract_symbols_in_line_range(
            math_file, 3, 5, "math_operations"
        )
        
        assert "math_operations.add" in symbols

    def test_non_python_files_ignored(self, temp_repo):
        """Test that non-Python files are ignored."""
        # Create non-Python files
        (temp_repo / "README.md").write_text("# Documentation")
        (temp_repo / "config.json").write_text("{}")
        (temp_repo / "test_file.txt").write_text("not python")
        
        test_selector = TestSelector(temp_repo)
        test_files = test_selector._discover_test_files()
        
        # Should only find Python test files
        test_file_names = [f.name for f in test_files]
        assert "README.md" not in test_file_names
        assert "config.json" not in test_file_names  
        assert "test_file.txt" not in test_file_names