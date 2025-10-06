"""LSP-powered code analyzer for precise patch targeting.

This module provides an analyzer that leverages Language Server Protocol (LSP)
capabilities through OpenCode's LSP integration to identify precise code ranges
for patch generation. This replaces manual line counting with semantic understanding
of code structure.

The analyzer can identify:
- Exact line ranges for functions, classes, and methods
- Docstring-only ranges (excluding function definitions)
- Function body ranges (excluding signatures and docstrings)
- Symbol locations and hierarchies
- Code construct boundaries

This solves the core issue where AI agents were including function definition
lines when they should only target docstrings or function bodies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from opencode_client import OpenCodeClient

logger = logging.getLogger(__name__)


class TargetType(str, Enum):
    """Types of code constructs that can be targeted for patches."""

    DOCSTRING = "docstring"
    FUNCTION_BODY = "function_body"
    ENTIRE_FUNCTION = "entire_function"
    CLASS_BODY = "class_body"
    ENTIRE_CLASS = "entire_class"
    METHOD_BODY = "method_body"
    ENTIRE_METHOD = "entire_method"
    VARIABLE = "variable"
    IMPORT = "import"
    CUSTOM_RANGE = "custom_range"


@dataclass
class CodeRange:
    """Represents a precise range of code lines with semantic information."""

    start_line: int  # Zero-indexed
    end_line: int  # Zero-indexed, inclusive
    target_type: TargetType
    symbol_name: str | None = None
    description: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] | None = None


@dataclass
class SymbolInfo:
    """Information about a code symbol with its location and structure."""

    name: str
    kind: str  # function, class, method, variable, etc.
    range: CodeRange
    docstring_range: CodeRange | None = None
    body_range: CodeRange | None = None
    children: list[SymbolInfo] | None = None
    metadata: dict[str, Any] | None = None


class LSPCodeAnalyzer:
    """LSP-powered code analyzer for precise patch targeting.

    This class uses OpenCode's LSP integration to perform semantic analysis
    of code files and identify precise line ranges for different code constructs.
    It provides methods to find docstring ranges, function bodies, and other
    code elements without relying on manual line counting.
    """

    def __init__(self, opencode_client: OpenCodeClient) -> None:
        """Initialize the LSP code analyzer.

        Args:
            opencode_client: OpenCode client with LSP capabilities.
        """
        self.opencode_client = opencode_client

    async def analyze_file(self, session_id: str, file_path: str) -> list[SymbolInfo]:
        """Analyze a file and return detailed symbol information.

        Uses LSP document symbols to build a comprehensive understanding
        of the file's structure with precise line ranges.

        Args:
            session_id: OpenCode session ID.
            file_path: Path to the file to analyze.

        Returns:
            List of symbol information with precise ranges.
        """
        try:
            # Get document symbols from LSP
            lsp_symbols = await self.opencode_client.get_document_symbols(
                session_id, file_path
            )

            if not lsp_symbols:
                logger.warning(f"No LSP symbols found for {file_path}")
                return []

            # Convert LSP symbols to our format with enhanced range analysis
            symbols = []
            for lsp_symbol in lsp_symbols:
                symbol_info = await self._parse_lsp_symbol(
                    session_id, file_path, lsp_symbol
                )
                if symbol_info:
                    symbols.append(symbol_info)

            logger.info(f"Analyzed {len(symbols)} symbols in {file_path}")
            return symbols

        except Exception as e:
            logger.error(f"Failed to analyze file {file_path}: {e}")
            return []

    async def find_docstring_range(
        self, session_id: str, file_path: str, function_name: str
    ) -> CodeRange | None:
        """Find the exact line range of a function's docstring.

        This is the key method that solves the original problem by identifying
        only the docstring lines, excluding the function definition.

        Args:
            session_id: OpenCode session ID.
            file_path: Path to the file containing the function.
            function_name: Name of the function to analyze.

        Returns:
            CodeRange for just the docstring lines, or None if not found.
        """
        try:
            symbols = await self.analyze_file(session_id, file_path)

            # Find the function symbol
            function_symbol = self._find_symbol_by_name(symbols, function_name)
            if not function_symbol:
                logger.warning(f"Function '{function_name}' not found in {file_path}")
                return None

            if function_symbol.docstring_range:
                logger.debug(
                    f"Found docstring range for {function_name}: lines {function_symbol.docstring_range.start_line}-{function_symbol.docstring_range.end_line}"
                )
                return function_symbol.docstring_range
            else:
                logger.debug(f"No docstring found for function '{function_name}'")
                return None

        except Exception as e:
            logger.error(
                f"Failed to find docstring range for {function_name} in {file_path}: {e}"
            )
            return None

    async def find_function_body_range(
        self, session_id: str, file_path: str, function_name: str
    ) -> CodeRange | None:
        """Find the exact line range of a function's body (excluding signature and docstring).

        Args:
            session_id: OpenCode session ID.
            file_path: Path to the file containing the function.
            function_name: Name of the function to analyze.

        Returns:
            CodeRange for just the function body lines, or None if not found.
        """
        try:
            symbols = await self.analyze_file(session_id, file_path)

            # Find the function symbol
            function_symbol = self._find_symbol_by_name(symbols, function_name)
            if not function_symbol:
                logger.warning(f"Function '{function_name}' not found in {file_path}")
                return None

            if function_symbol.body_range:
                logger.debug(
                    f"Found body range for {function_name}: lines {function_symbol.body_range.start_line}-{function_symbol.body_range.end_line}"
                )
                return function_symbol.body_range
            else:
                logger.debug(f"No body range computed for function '{function_name}'")
                return None

        except Exception as e:
            logger.error(
                f"Failed to find body range for {function_name} in {file_path}: {e}"
            )
            return None

    async def find_symbol_by_problem_description(
        self, session_id: str, file_path: str, problem_description: str
    ) -> list[tuple[SymbolInfo, TargetType]]:
        """Find symbols that are likely related to a problem description.

        Uses semantic analysis to identify which functions, classes, or other
        code constructs are most likely to need modification based on the
        problem description.

        Args:
            session_id: OpenCode session ID.
            file_path: Path to the file to analyze.
            problem_description: Description of the issue to address.

        Returns:
            List of (symbol, target_type) tuples ranked by relevance.
        """
        try:
            symbols = await self.analyze_file(session_id, file_path)

            # Analyze the problem description for clues
            problem_lower = problem_description.lower()
            relevant_symbols = []

            for symbol in symbols:
                relevance_score = 0.0
                suggested_target = TargetType.ENTIRE_FUNCTION

                # Check if symbol name appears in problem description
                if symbol.name.lower() in problem_lower:
                    relevance_score += 1.0

                # Check for docstring-related keywords
                if any(
                    keyword in problem_lower
                    for keyword in [
                        "docstring",
                        "documentation",
                        "doc",
                        "comment",
                        "description",
                        "document",
                    ]
                ):
                    if symbol.docstring_range:
                        suggested_target = TargetType.DOCSTRING
                        relevance_score += 0.5

                # Check for implementation-related keywords
                elif any(
                    keyword in problem_lower
                    for keyword in [
                        "implementation",
                        "logic",
                        "algorithm",
                        "bug",
                        "fix",
                        "error",
                        "incorrect",
                        "wrong",
                    ]
                ):
                    suggested_target = TargetType.FUNCTION_BODY
                    relevance_score += 0.3

                # Boost score for functions vs other symbol types
                if symbol.kind in ["function", "method"]:
                    relevance_score += 0.2
                elif symbol.kind == "class":
                    suggested_target = TargetType.CLASS_BODY

                if relevance_score > 0.1:  # Only include reasonably relevant symbols
                    relevant_symbols.append((symbol, suggested_target))

            # Sort by relevance score
            relevant_symbols.sort(
                key=lambda x: x[0].metadata.get("relevance_score", 0.0), reverse=True
            )

            logger.info(
                f"Found {len(relevant_symbols)} relevant symbols for problem description"
            )
            return relevant_symbols

        except Exception as e:
            logger.error(
                f"Failed to find symbols for problem description in {file_path}: {e}"
            )
            return []

    async def _parse_lsp_symbol(
        self, session_id: str, file_path: str, lsp_symbol: dict[str, Any]
    ) -> SymbolInfo | None:
        """Parse an LSP symbol response into our SymbolInfo format.

        Args:
            session_id: OpenCode session ID.
            file_path: Path to the file containing the symbol.
            lsp_symbol: Raw LSP symbol data.

        Returns:
            Parsed SymbolInfo with computed ranges, or None if parsing fails.
        """
        try:
            name = lsp_symbol.get("name", "")
            kind = lsp_symbol.get("kind", "unknown")

            # Extract range information from LSP response
            # LSP ranges are typically in format: {"start": {"line": 0, "character": 0}, "end": {"line": 5, "character": 10}}
            lsp_range = lsp_symbol.get("range", {})
            start_line = lsp_range.get("start", {}).get("line", 0)
            end_line = lsp_range.get("end", {}).get("line", 0)

            # Create main symbol range
            main_range = CodeRange(
                start_line=start_line,
                end_line=end_line,
                target_type=(
                    TargetType.ENTIRE_FUNCTION
                    if kind == "function"
                    else TargetType.CUSTOM_RANGE
                ),
                symbol_name=name,
                description=f"{kind} {name}",
                metadata=lsp_symbol,
            )

            # For functions and methods, try to identify docstring and body ranges
            docstring_range = None
            body_range = None

            if kind in ["function", "method"]:
                docstring_range = await self._find_docstring_in_function(
                    session_id, file_path, name, start_line, end_line
                )
                body_range = await self._find_body_in_function(
                    session_id, file_path, name, start_line, end_line, docstring_range
                )

            # Parse children if present
            children = []
            if "children" in lsp_symbol:
                for child_symbol in lsp_symbol["children"]:
                    child_info = await self._parse_lsp_symbol(
                        session_id, file_path, child_symbol
                    )
                    if child_info:
                        children.append(child_info)

            return SymbolInfo(
                name=name,
                kind=kind,
                range=main_range,
                docstring_range=docstring_range,
                body_range=body_range,
                children=children if children else None,
                metadata=lsp_symbol,
            )

        except Exception as e:
            logger.error(f"Failed to parse LSP symbol: {e}")
            return None

    async def _find_docstring_in_function(
        self,
        session_id: str,
        file_path: str,
        function_name: str,
        func_start: int,
        func_end: int,
    ) -> CodeRange | None:
        """Find the docstring range within a function.

        This uses heuristics and possibly LSP hover info to identify
        the exact lines containing just the docstring.
        """
        try:
            # Read the file content to analyze the function
            file_content = await self.opencode_client.read_file(session_id, file_path)
            if not file_content:
                return None

            lines = file_content.split("\n")

            # Look for docstring patterns starting after the function definition
            docstring_start = None
            docstring_end = None
            in_docstring = False
            quote_style = None

            # Start looking from the line after the function definition
            search_start = func_start + 1

            for i in range(search_start, min(func_end + 1, len(lines))):
                line = lines[i].strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Look for docstring start
                if not in_docstring:
                    if line.startswith('"""') or line.startswith("'''"):
                        quote_style = line[:3]
                        docstring_start = i

                        # Check if it's a single-line docstring
                        if line.endswith(quote_style) and len(line) > 6:
                            docstring_end = i
                            break
                        else:
                            in_docstring = True
                    elif line.startswith('"') or line.startswith("'"):
                        # Single-line string as docstring
                        quote_char = line[0]
                        if line.endswith(quote_char) and len(line) > 2:
                            docstring_start = docstring_end = i
                            break
                    else:
                        # Hit non-docstring code, no docstring present
                        break

                # Look for docstring end
                elif in_docstring and quote_style and line.endswith(quote_style):
                    docstring_end = i
                    break

            if docstring_start is not None and docstring_end is not None:
                return CodeRange(
                    start_line=docstring_start,
                    end_line=docstring_end,
                    target_type=TargetType.DOCSTRING,
                    symbol_name=function_name,
                    description=f"Docstring for {function_name}",
                    confidence=0.9,
                )

        except Exception as e:
            logger.error(f"Failed to find docstring in function {function_name}: {e}")

        return None

    async def _find_body_in_function(
        self,
        session_id: str,
        file_path: str,
        function_name: str,
        func_start: int,
        func_end: int,
        docstring_range: CodeRange | None,
    ) -> CodeRange | None:
        """Find the body range within a function (excluding signature and docstring)."""
        try:
            # Start after function definition (and docstring if present)
            body_start = func_start + 1
            if docstring_range:
                body_start = docstring_range.end_line + 1

            # End at the function end
            body_end = func_end

            # Validate the range
            if body_start <= body_end:
                return CodeRange(
                    start_line=body_start,
                    end_line=body_end,
                    target_type=TargetType.FUNCTION_BODY,
                    symbol_name=function_name,
                    description=f"Body of {function_name}",
                    confidence=0.8,
                )

        except Exception as e:
            logger.error(f"Failed to find body in function {function_name}: {e}")

        return None

    def _find_symbol_by_name(
        self, symbols: list[SymbolInfo], name: str
    ) -> SymbolInfo | None:
        """Find a symbol by name in the symbol list (including children)."""
        for symbol in symbols:
            if symbol.name == name:
                return symbol
            if symbol.children:
                child_result = self._find_symbol_by_name(symbol.children, name)
                if child_result:
                    return child_result
        return None
