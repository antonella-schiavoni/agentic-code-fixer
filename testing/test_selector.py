"""Smart test selector that analyzes patches and runs only relevant tests.

This module implements intelligent test selection by:
1. Analyzing applied patches to identify modified symbols (functions/classes)  
2. Building and maintaining an index of which tests import/use which symbols
3. Selecting only the tests that are relevant to the patched code
4. Falling back to full test suite when needed for safety

This approach dramatically reduces test execution time while maintaining confidence
that patches don't break functionality.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from core.types import PatchCandidate

logger = logging.getLogger(__name__)


class TestSelector:
    """Smart test selector that identifies relevant tests for patch validation."""
    
    def __init__(
        self, 
        repository_path: str | Path,
        cache_dir: str | Path = ".cache",
        test_patterns: list[str] | None = None
    ):
        """Initialize the test selector.
        
        Args:
            repository_path: Path to the repository root
            cache_dir: Directory for caching test symbol index 
            test_patterns: Patterns for discovering test files (defaults to pytest patterns)
        """
        self.repository_path = Path(repository_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Default pytest test file patterns
        self.test_patterns = test_patterns or [
            "test_*.py", 
            "*_test.py", 
            "tests/*.py",
            "tests/**/test_*.py", 
            "tests/**/*_test.py"
        ]
        
        self.index_cache_file = self.cache_dir / "test_symbol_index.json"
        self.index_meta_file = self.cache_dir / "test_symbol_index_meta.json"
        
        # In-memory cache of the test symbol index
        self._symbol_index: Dict[str, Set[str]] | None = None
        self._last_index_time = 0.0

    def select_tests_for_patches(self, patches: list[PatchCandidate]) -> list[str]:
        """Select specific test functions that are relevant to the given patches.
        
        Args:
            patches: List of patches to analyze
            
        Returns:
            List of specific test identifiers (e.g., 'tests/test_file.py::test_function') 
            that should be run. Returns empty list if all tests should be run (fallback case).
        """
        try:
            # Step 1: Extract modified symbols from patches
            modified_symbols = self._extract_modified_symbols(patches)
            
            if not modified_symbols:
                logger.info("No symbols extracted from patches, running all tests")
                return []
                
            logger.info(f"Extracted {len(modified_symbols)} modified symbols: {modified_symbols}")
            
            # Step 2: Get or build the test symbol index
            symbol_index = self._get_test_symbol_index()
            
            if not symbol_index:
                logger.warning("Failed to build test symbol index, running all tests")
                return []
            
            # Step 3: Find specific test functions that reference the modified symbols
            relevant_tests = set()
            for symbol in modified_symbols:
                if symbol in symbol_index:
                    relevant_tests.update(symbol_index[symbol])
                    
            # Also check for partial matches (e.g., class methods, different import paths)
            for symbol in modified_symbols:
                for indexed_symbol, test_identifiers in symbol_index.items():
                    if self._symbols_related(symbol, indexed_symbol):
                        relevant_tests.update(test_identifiers)
                        
            # Check for function name matches across different module paths
            for symbol in modified_symbols:
                symbol_parts = symbol.split('.')
                if len(symbol_parts) >= 2:
                    function_name = symbol_parts[-1]  # Get the function name
                    for indexed_symbol, test_identifiers in symbol_index.items():
                        indexed_parts = indexed_symbol.split('.')
                        # Match if the function names are the same
                        if (len(indexed_parts) >= 1 and 
                            indexed_parts[-1] == function_name):
                            relevant_tests.update(test_identifiers)
                            logger.debug(f"Matched {symbol} with {indexed_symbol} via function name")
            
            # FALLBACK: If no symbols extracted from patches, try filename-based matching
            # This handles cases where patch content analysis fails but we can still
            # infer the target function from the file being modified
            if not modified_symbols:
                logger.debug("No symbols extracted from patches, trying filename-based fallback")
                for patch in patches:
                    try:
                        file_path = Path(patch.file_path)
                        if file_path.suffix == '.py':
                            module_name = self._file_path_to_module_name(file_path)
                            
                            # Look for any symbols in the index that belong to this module
                            for indexed_symbol, test_identifiers in symbol_index.items():
                                if indexed_symbol.startswith(f"{module_name}.") or indexed_symbol in symbol_index:
                                    relevant_tests.update(test_identifiers)
                                    logger.debug(f"Fallback: matched {indexed_symbol} for module {module_name}")
                                    
                    except Exception as e:
                        logger.debug(f"Error in filename-based fallback for patch {patch.id}: {e}")
            
            relevant_test_list = sorted(relevant_tests)
            
            if not relevant_test_list:
                logger.info("No relevant tests found, running all tests as safety fallback")
                return []
                
            # Count total test functions for better reporting
            total_test_functions = self._count_total_test_functions()
            percentage = (len(relevant_test_list) / total_test_functions * 100) if total_test_functions > 0 else 0
            
            logger.info(
                f"Selected {len(relevant_test_list)}/{total_test_functions} test functions ({percentage:.1f}%) "
                f"because they reference {len(modified_symbols)} modified symbols"
            )
            
            return relevant_test_list
            
        except Exception as e:
            logger.error(f"Error selecting tests for patches: {e}", exc_info=True)
            logger.info("Falling back to running all tests due to error")
            return []

    def _extract_modified_symbols(self, patches: list[PatchCandidate]) -> Set[str]:
        """Extract fully-qualified names of functions/classes modified by patches.
        
        Args:
            patches: List of patches to analyze
            
        Returns:
            Set of qualified symbol names like "module.function" or "module.Class.method"
        """
        modified_symbols = set()
        
        for patch in patches:
            try:
                # Get the file path and convert to module name
                file_path = Path(patch.file_path)
                if file_path.suffix != '.py':
                    continue
                    
                module_name = self._file_path_to_module_name(file_path)
                
                # Parse the patch content to find function/class definitions
                symbols_in_patch = self._parse_symbols_from_content(patch.content, module_name)
                modified_symbols.update(symbols_in_patch)
                
                # Also try to analyze the target file to see what symbols are being modified
                target_file = self.repository_path / file_path
                if target_file.exists():
                    symbols_in_range = self._extract_symbols_in_line_range(
                        target_file, patch.line_start, patch.line_end, module_name
                    )
                    modified_symbols.update(symbols_in_range)
                    
            except Exception as e:
                logger.debug(f"Error extracting symbols from patch {patch.id}: {e}")
                continue
                
        return modified_symbols

    def _parse_symbols_from_content(self, content: str, module_name: str) -> Set[str]:
        """Parse function/class definitions from patch content.
        
        Args:
            content: The patch content (code)
            module_name: Module name to prefix symbols with
            
        Returns:
            Set of qualified symbol names
        """
        symbols = set()
        
        try:
            # Try to parse the content as Python code
            # Note: patch content might be a fragment, so we may need to handle SyntaxError
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    symbols.add(f"{module_name}.{node.name}")
                elif isinstance(node, ast.ClassDef):
                    symbols.add(f"{module_name}.{node.name}")
                    # Also add methods within the class
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            symbols.add(f"{module_name}.{node.name}.{item.name}")
                            
        except SyntaxError:
            # Content might be a fragment, try regex patterns as fallback
            symbols.update(self._extract_symbols_with_regex(content, module_name))
            
        return symbols

    def _extract_symbols_with_regex(self, content: str, module_name: str) -> Set[str]:
        """Extract symbols using regex patterns as fallback when AST parsing fails."""
        symbols = set()
        
        # Function definitions
        func_pattern = r'^[ \t]*def\s+(\w+)\s*\('
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            symbols.add(f"{module_name}.{match.group(1)}")
            
        # Class definitions  
        class_pattern = r'^[ \t]*class\s+(\w+)(?:\([^)]*\))?\s*:'
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            symbols.add(f"{module_name}.{match.group(1)}")
            
        return symbols

    def _extract_symbols_in_line_range(
        self, 
        file_path: Path, 
        line_start: int, 
        line_end: int, 
        module_name: str
    ) -> Set[str]:
        """Extract symbols that contain the specified line range in a file.
        
        Args:
            file_path: Path to the Python file
            line_start: Start line (0-indexed)
            line_end: End line (0-indexed) 
            module_name: Module name for the file
            
        Returns:
            Set of symbols whose definitions contain the line range
        """
        symbols = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            lines = content.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    # Check if the patch line range overlaps with this symbol's definition
                    if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                        # Convert to 0-indexed
                        symbol_start = node.lineno - 1
                        symbol_end = getattr(node, 'end_lineno', node.lineno) - 1
                        
                        # FIXED: Extend symbol range to catch patches targeting function boundaries
                        # Many patches target the line immediately after a function (like return statements)
                        # or empty lines between functions. Extend by 2 lines to catch these cases.
                        extended_symbol_end = symbol_end + 2
                        
                        # Check if patch range overlaps with extended symbol range
                        if self._ranges_overlap(line_start, line_end, symbol_start, extended_symbol_end):
                            if isinstance(node, ast.FunctionDef):
                                # Check if this is a method inside a class
                                parent_class = self._find_parent_class(tree, node)
                                if parent_class:
                                    symbols.add(f"{module_name}.{parent_class}.{node.name}")
                                else:
                                    symbols.add(f"{module_name}.{node.name}")
                                    logger.debug(
                                        f"Matched function {node.name} (lines {symbol_start}-{symbol_end}, "
                                        f"extended to {extended_symbol_end}) with patch (lines {line_start}-{line_end})"
                                    )
                            elif isinstance(node, ast.ClassDef):
                                symbols.add(f"{module_name}.{node.name}")
                                
        except Exception as e:
            logger.debug(f"Error extracting symbols from {file_path}: {e}")
            
        return symbols

    def _find_parent_class(self, tree: ast.AST, target_node: ast.FunctionDef) -> str | None:
        """Find the parent class name for a function definition."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if target_node in ast.walk(node):
                    return node.name
        return None

    def _ranges_overlap(self, start1: int, end1: int, start2: int, end2: int) -> bool:
        """Check if two line ranges overlap."""
        return max(start1, start2) <= min(end1, end2)

    def _file_path_to_module_name(self, file_path: Path) -> str:
        """Convert a file path to a Python module name.
        
        Args:
            file_path: Path relative to repository root
            
        Returns:
            Module name like "pkg.subpkg.module"
        """
        # Remove .py extension and convert path separators to dots
        module_parts = file_path.with_suffix('').parts
        
        # Remove common prefixes like 'src/' if they exist
        if module_parts and module_parts[0] in ('src', 'lib'):
            module_parts = module_parts[1:]
            
        return '.'.join(module_parts)

    def _get_test_symbol_index(self) -> Dict[str, Set[str]]:
        """Get or build the test symbol index.
        
        Returns:
            Dictionary mapping symbols to test files that import them
        """
        if self._symbol_index and self._is_index_fresh():
            return self._symbol_index
            
        # Try to load from cache
        if self._load_index_from_cache():
            return self._symbol_index
            
        # Build new index
        logger.info("Building test symbol index...")
        start_time = time.time()
        
        self._symbol_index = self._build_test_symbol_index()
        self._last_index_time = time.time()
        
        # Save to cache
        self._save_index_to_cache()
        
        elapsed = time.time() - start_time
        logger.info(
            f"Built test symbol index with {len(self._symbol_index)} symbols "
            f"in {elapsed:.2f}s"
        )
        
        return self._symbol_index

    def _is_index_fresh(self) -> bool:
        """Check if the current index is still fresh."""
        if not self._symbol_index or not self.index_meta_file.exists():
            return False
            
        try:
            with open(self.index_meta_file, 'r') as f:
                meta = json.load(f)
                
            # Check if any test files have been modified since the index was built
            for test_file in self._discover_test_files():
                file_mtime = test_file.stat().st_mtime
                if file_mtime > meta.get('build_time', 0):
                    return False
                    
            return True
            
        except Exception:
            return False

    def _load_index_from_cache(self) -> bool:
        """Load the symbol index from cache.
        
        Returns:
            True if successfully loaded, False otherwise
        """
        if not self.index_cache_file.exists() or not self._is_index_fresh():
            return False
            
        try:
            with open(self.index_cache_file, 'r') as f:
                data = json.load(f)
                
            # Convert sets back from lists
            self._symbol_index = {
                symbol: set(test_files) 
                for symbol, test_files in data.items()
            }
            
            logger.info(f"Loaded test symbol index with {len(self._symbol_index)} symbols from cache")
            return True
            
        except Exception as e:
            logger.debug(f"Failed to load index from cache: {e}")
            return False

    def _save_index_to_cache(self) -> None:
        """Save the symbol index to cache."""
        if not self._symbol_index:
            return
            
        try:
            # Convert sets to lists for JSON serialization
            data = {
                symbol: sorted(test_files)
                for symbol, test_files in self._symbol_index.items()
            }
            
            with open(self.index_cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            # Save metadata
            meta = {
                'build_time': self._last_index_time,
                'repository_path': str(self.repository_path),
                'test_file_count': self._count_total_test_files()
            }
            
            with open(self.index_meta_file, 'w') as f:
                json.dump(meta, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save index to cache: {e}")

    def _build_test_symbol_index(self) -> Dict[str, Set[str]]:
        """Build a fresh test symbol index by analyzing all test files.
        
        Returns:
            Dictionary mapping symbols to specific test functions that import/use them
        """
        symbol_index = defaultdict(set)
        
        for test_file in self._discover_test_files():
            try:
                imports, test_functions = self._analyze_test_file_imports_and_functions(test_file)
                
                # Convert to relative path for storage
                relative_path = test_file.relative_to(self.repository_path)
                
                # Smart mapping: only map symbols to test functions that likely test them
                for symbol in imports:
                    for test_func in test_functions:
                        if self._test_function_likely_tests_symbol(test_func, symbol):
                            # Create pytest-compatible test identifier
                            test_identifier = f"{relative_path}::{test_func}"
                            symbol_index[symbol].add(test_identifier)
                    
            except Exception as e:
                logger.debug(f"Error analyzing test file {test_file}: {e}")
                continue
                
        # Convert defaultdict to regular dict
        return {symbol: file_set for symbol, file_set in symbol_index.items()}

    def _discover_test_files(self) -> List[Path]:
        """Discover all test files in the repository.
        
        Returns:
            List of test file paths
        """
        test_files = []
        
        for pattern in self.test_patterns:
            test_files.extend(self.repository_path.glob(pattern))
            
        # Remove duplicates and ensure files exist
        unique_files = []
        seen = set()
        
        for file_path in test_files:
            if file_path.is_file() and file_path.suffix == '.py':
                abs_path = file_path.resolve()
                if abs_path not in seen:
                    seen.add(abs_path)
                    unique_files.append(file_path)
                    
        return unique_files

    def _analyze_test_file_imports_and_functions(self, test_file: Path) -> tuple[Set[str], Set[str]]:
        """Analyze a test file to extract imported symbols and test function names.
        
        Args:
            test_file: Path to the test file
            
        Returns:
            Tuple of (imported symbol names, test function names)
        """
        imports = set()
        test_functions = set()
        
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                        # Add all sub-attributes that might be used
                        imports.update(self._generate_submodule_names(alias.name))
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            if alias.name != '*':
                                qualified_name = f"{node.module}.{alias.name}"
                                imports.add(qualified_name)
                                imports.add(alias.name)  # Also add unqualified name
                                
                elif isinstance(node, ast.Attribute):
                    # Capture attribute access patterns like obj.method_name
                    if isinstance(node.value, ast.Name):
                        attr_name = f"{node.value.id}.{node.attr}"
                        imports.add(attr_name)
                        
                elif isinstance(node, ast.FunctionDef):
                    # Extract test function names (functions starting with 'test_')
                    if node.name.startswith('test_'):
                        test_functions.add(node.name)
                        
        except Exception as e:
            logger.debug(f"Error parsing test file {test_file}: {e}")
            
        return imports, test_functions
    
    def _analyze_test_file_imports(self, test_file: Path) -> Set[str]:
        """Analyze a test file to extract all imported symbols.
        
        Args:
            test_file: Path to the test file
            
        Returns:
            Set of imported symbol names
        """
        imports, _ = self._analyze_test_file_imports_and_functions(test_file)
        return imports

    def _generate_submodule_names(self, module_name: str) -> Set[str]:
        """Generate potential submodule names from a module import.
        
        Args:
            module_name: Full module name like "pkg.subpkg.module"
            
        Returns:
            Set of potential symbol names
        """
        submodules = set()
        parts = module_name.split('.')
        
        # Generate all possible combinations
        for i in range(1, len(parts) + 1):
            submodules.add('.'.join(parts[:i]))
            
        return submodules

    def _symbols_related(self, symbol1: str, symbol2: str) -> bool:
        """Check if two symbols are related (one is a subset of the other).
        
        Args:
            symbol1: First symbol name
            symbol2: Second symbol name
            
        Returns:
            True if symbols are related
        """
        # Check if one symbol is a prefix of another (class.method relationship)
        return symbol1.startswith(symbol2 + '.') or symbol2.startswith(symbol1 + '.')

    def _count_total_test_files(self) -> int:
        """Count total number of test files in the repository."""
        return len(self._discover_test_files())
    
    def _count_total_test_functions(self) -> int:
        """Count total number of test functions in the repository."""
        total = 0
        for test_file in self._discover_test_files():
            try:
                _, test_functions = self._analyze_test_file_imports_and_functions(test_file)
                total += len(test_functions)
            except Exception as e:
                logger.debug(f"Error counting test functions in {test_file}: {e}")
        return total
    
    def _test_function_likely_tests_symbol(self, test_function_name: str, symbol: str) -> bool:
        """Determine if a test function likely tests a specific symbol based on naming conventions.
        
        Args:
            test_function_name: Name of the test function (e.g., 'test_register_payment')
            symbol: Symbol name (e.g., 'src.dictionaries.register_payment' or 'register_payment')
            
        Returns:
            True if the test function likely tests this symbol
        """
        # Extract the base function name from the symbol
        symbol_parts = symbol.split('.')
        symbol_function_name = symbol_parts[-1]  # Get the last part (function name)
        
        # Common test naming patterns:
        # 1. test_function_name -> function_name
        # 2. test_FunctionName -> FunctionName  
        # 3. testFunctionName -> functionName
        
        # Remove 'test_' prefix if present
        if test_function_name.startswith('test_'):
            test_base_name = test_function_name[5:]  # Remove 'test_' prefix
            return test_base_name == symbol_function_name
            
        # Handle camelCase: testFunctionName -> functionName
        if test_function_name.startswith('test') and len(test_function_name) > 4:
            test_base_name = test_function_name[4].lower() + test_function_name[5:]  # testFunction -> function
            return test_base_name == symbol_function_name
            
        # Fallback: partial string matching for complex cases
        return symbol_function_name.lower() in test_function_name.lower()

    def create_test_manifest(self, test_identifiers: list[str], manifest_path: Path | str | None = None) -> Path:
        """Create a test manifest file for pytest to consume.
        
        Args:
            test_identifiers: List of test identifiers (e.g., 'tests/test_file.py::test_function') to include
            manifest_path: Optional path for the manifest file
            
        Returns:
            Path to the created manifest file
        """
        if manifest_path is None:
            manifest_path = self.repository_path / ".agentic_selected_tests"
        else:
            manifest_path = Path(manifest_path)
            
        try:
            with open(manifest_path, 'w') as f:
                for test_identifier in test_identifiers:
                    f.write(f"{test_identifier}\n")
                    
            logger.info(f"Created test manifest at {manifest_path} with {len(test_identifiers)} test functions")
            return manifest_path
            
        except Exception as e:
            logger.error(f"Failed to create test manifest: {e}")
            raise

    def cleanup_manifest(self, manifest_path: Path | str | None = None) -> None:
        """Clean up the test manifest file.
        
        Args:
            manifest_path: Optional path to the manifest file to clean up
        """
        if manifest_path is None:
            manifest_path = self.repository_path / ".agentic_selected_tests"
        else:
            manifest_path = Path(manifest_path)
            
        if manifest_path.exists():
            try:
                manifest_path.unlink()
                logger.debug(f"Cleaned up test manifest at {manifest_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up test manifest: {e}")