"""Intelligent code indexing system for semantic code analysis and retrieval.

This module provides comprehensive code indexing capabilities that enable semantic
search and analysis of codebases. It processes source code files, extracts
meaningful information, generates embeddings for similarity search, and stores
everything in a vector database for efficient retrieval.

The indexer supports multiple programming languages and uses advanced techniques
for code parsing, function extraction, and dependency analysis. This enables
AI agents to find relevant code contexts when generating patches.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path

from indexing.vector_store import VectorStore

from core.config import VectorDBConfig
from core.types import CodeContext

logger = logging.getLogger(__name__)


class CodeIndexer:
    """Advanced code indexing system for semantic analysis and intelligent retrieval.

    This class provides comprehensive codebase indexing capabilities that enable
    semantic search and contextual code analysis. It processes source files across
    multiple programming languages, extracts structural information, generates
    vector embeddings, and stores everything for efficient similarity-based retrieval.

    The indexer uses sophisticated parsing techniques to extract:
    - Function and class definitions
    - Import statements and dependencies
    - Code structure and relationships
    - Semantic embeddings for similarity search

    This enables AI agents to find relevant code contexts when analyzing problems
    and generating patch solutions.

    Attributes:
        config: Vector database configuration for embedding and storage settings.
        vector_store: Vector database instance for embedding storage and retrieval.
        language_extensions: Mapping of programming languages to file extensions.
    """

    def __init__(self, config: VectorDBConfig) -> None:
        """Initialize the code indexer with vector database configuration.

        Args:
            config: Vector database configuration including embedding model,
                storage settings, and chunking parameters.
        """
        self.config = config
        self.vector_store = VectorStore(config)

        # Language-specific file extensions
        self.language_extensions = {
            "python": [".py", ".pyx", ".pyi"],
            "javascript": [".js", ".jsx", ".mjs"],
            "typescript": [".ts", ".tsx"],
            "java": [".java"],
            "cpp": [".cpp", ".cc", ".cxx", ".hpp", ".h"],
            "c": [".c", ".h"],
            "rust": [".rs"],
            "go": [".go"],
            "ruby": [".rb"],
            "php": [".php"],
            "swift": [".swift"],
            "kotlin": [".kt", ".kts"],
            "scala": [".scala"],
            "clojure": [".clj", ".cljs"],
            "haskell": [".hs"],
            "erlang": [".erl"],
            "elixir": [".ex", ".exs"],
        }

    def index_repository(
        self,
        repo_path: str | Path,
        exclude_patterns: list[str] | None = None,
        target_files: list[str] | None = None,
    ) -> list[CodeContext]:
        """Index all relevant files in a repository."""
        repo_path = Path(repo_path)
        exclude_patterns = exclude_patterns or []

        if not repo_path.exists():
            raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

        logger.info(f"Starting repository indexing: {repo_path}")

        # Find all code files
        code_files = self._find_code_files(repo_path, exclude_patterns, target_files)
        logger.info(f"Found {len(code_files)} code files to index")

        # Process files and create contexts
        contexts = []
        for file_path in code_files:
            try:
                file_contexts = self._process_file(file_path)
                contexts.extend(file_contexts)
            except Exception as e:
                logger.warning(f"Failed to process file {file_path}: {e}")

        logger.info(f"Created {len(contexts)} code contexts")

        # Add contexts to vector store
        if contexts:
            self.vector_store.add_code_context(contexts)

        return contexts

    def search_relevant_context(
        self,
        problem_description: str,
        top_k: int = 10, #TODO: We may need to increase this number
        file_filter: str | None = None,
    ) -> list[CodeContext]:
        """Search for code contexts relevant to a problem description."""
        results = self.vector_store.search_similar_contexts(
            query=problem_description,
            top_k=top_k,
            file_filter=file_filter
        )

        # Extract just the contexts (without scores)
        return [context for context, _ in results]

    def get_file_context(self, file_path: str) -> list[CodeContext]:
        """Get all contexts for a specific file."""
        return self.vector_store.get_context_by_file(file_path)

    def _find_code_files(
        self,
        repo_path: Path,
        exclude_patterns: list[str],
        target_files: list[str] | None = None,
    ) -> list[Path]:
        """Find all code files in the repository."""
        #TODO: Remove target_files, we should index the entire repository, not just specific files
        if target_files:
            # Use specific target files
            files = []
            for target in target_files:
                target_path = repo_path / target
                if target_path.exists() and target_path.is_file():
                    files.append(target_path)
                else:
                    logger.warning(f"Target file not found: {target_path}")
            return files

        # Find all code files recursively
        code_files = []
        all_extensions = set()
        for exts in self.language_extensions.values():
            all_extensions.update(exts)

        for file_path in repo_path.rglob("*"):
            if not file_path.is_file():
                continue

            # Check if file should be excluded
            if self._should_exclude_file(file_path, repo_path, exclude_patterns):
                continue

            # Check if it's a code file
            if file_path.suffix.lower() in all_extensions:
                code_files.append(file_path)

        return code_files

    def _should_exclude_file(
        self,
        file_path: Path,
        repo_path: Path,
        exclude_patterns: list[str]
    ) -> bool:
        """Check if a file should be excluded based on patterns."""
        relative_path = file_path.relative_to(repo_path)
        path_str = str(relative_path)

        for pattern in exclude_patterns:
            if pattern in path_str or file_path.match(pattern):
                return True

        return False

    def _process_file(self, file_path: Path) -> list[CodeContext]:
        """Process a single file and create code contexts."""
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Failed to read file {file_path}: {e}")
            return []

        if not content.strip():
            return []

        # Determine language
        language = self._detect_language(file_path)

        # Split content into chunks if too large
        if len(content) <= self.config.chunk_size:
            # Small file - create single context
            context = CodeContext(
                file_path=str(file_path),
                content=content,
                language=language,
                relevant_functions=self._extract_functions(content, language),
                dependencies=self._extract_dependencies(content, language),
            )
            return [context]
        else:
            # Large file - split into chunks
            return self._chunk_file_content(file_path, content, language)

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        #TODO: If required, we could use a library to detect the programming language in a more sophisticated way
        suffix = file_path.suffix.lower()

        for language, extensions in self.language_extensions.items():
            if suffix in extensions:
                return language

        return "unknown"

    def _chunk_file_content(
        self,
        file_path: Path,
        content: str,
        language: str
    ) -> list[CodeContext]:
        """Split large file content into smaller chunks."""
        contexts = []
        lines = content.split("\n")

        chunk_lines = []
        current_size = 0

        for line in lines:
            line_size = len(line) + 1  # +1 for newline

            if current_size + line_size > self.config.chunk_size and chunk_lines:
                # Create context for current chunk
                chunk_content = "\n".join(chunk_lines)
                context = CodeContext(
                    file_path=str(file_path),
                    content=chunk_content,
                    language=language,
                    relevant_functions=self._extract_functions(chunk_content, language),
                    dependencies=self._extract_dependencies(chunk_content, language),
                )
                contexts.append(context)

                # Start new chunk with overlap
                overlap_lines = max(0, len(chunk_lines) - self.config.chunk_overlap)
                chunk_lines = chunk_lines[overlap_lines:] + [line]
                current_size = sum(len(l) + 1 for l in chunk_lines)
            else:
                chunk_lines.append(line)
                current_size += line_size

        # Add final chunk if any content remains
        if chunk_lines:
            chunk_content = "\n".join(chunk_lines)
            context = CodeContext(
                file_path=str(file_path),
                content=chunk_content,
                language=language,
                relevant_functions=self._extract_functions(chunk_content, language),
                dependencies=self._extract_dependencies(chunk_content, language),
            )
            contexts.append(context)

        return contexts

    def _extract_functions(self, content: str, language: str) -> list[str]:
        """Extract function names from code content."""
        functions = []

        #TODO: Improve this 
        if language == "python":
            functions = self._extract_python_functions(content)
        elif language in ["javascript", "typescript"]:
            functions = self._extract_js_functions(content)
        elif language == "java":
            functions = self._extract_java_functions(content)
        # Add more language-specific extractors as needed

        return functions

    def _extract_python_functions(self, content: str) -> list[str]:
        """Extract Python function and class names using AST."""
        functions = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    functions.append(node.name)
        except SyntaxError:
            # Fall back to regex if AST parsing fails
            functions = re.findall(r"def\s+(\w+)|class\s+(\w+)", content)
            functions = [f[0] or f[1] for f in functions]

        return functions

    def _extract_js_functions(self, content: str) -> list[str]:
        """Extract JavaScript/TypeScript function names using regex."""
        patterns = [
            r"function\s+(\w+)",
            r"const\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)\s*=>\s*{|\([^)]*\)\s*{|function)",
            r"(\w+)\s*:\s*(?:async\s+)?function",
            r"class\s+(\w+)",
        ]

        functions = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            functions.extend(matches)

        return functions

    def _extract_java_functions(self, content: str) -> list[str]:
        """Extract Java method and class names using regex."""
        patterns = [
            r"class\s+(\w+)",
            r"interface\s+(\w+)",
            r"(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*{",
        ]

        functions = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            functions.extend(matches)

        return functions

    def _extract_dependencies(self, content: str, language: str) -> list[str]:
        """Extract import/dependency information from code."""
        dependencies = []

        if language == "python":
            dependencies = self._extract_python_imports(content)
        elif language in ["javascript", "typescript"]:
            dependencies = self._extract_js_imports(content)
        elif language == "java":
            dependencies = self._extract_java_imports(content)

        return dependencies

    def _extract_python_imports(self, content: str) -> list[str]:
        """Extract Python import statements."""
        imports = []
        patterns = [
            r"import\s+([^\s,]+)",
            r"from\s+([^\s]+)\s+import",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content)
            imports.extend(matches)

        return imports

    def _extract_js_imports(self, content: str) -> list[str]:
        """Extract JavaScript/TypeScript import statements."""
        patterns = [
            r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]",
            r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
        ]

        imports = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            imports.extend(matches)

        return imports

    def _extract_java_imports(self, content: str) -> list[str]:
        """Extract Java import statements."""
        pattern = r"import\s+(?:static\s+)?([^;]+);"
        imports = re.findall(pattern, content)
        return imports