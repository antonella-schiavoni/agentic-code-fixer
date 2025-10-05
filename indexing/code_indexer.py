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

from core.config import OpenCodeConfig, VectorDBConfig
from core.types import CodeContext
from indexing.vector_store import VectorStore

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

    def __init__(
        self, config: VectorDBConfig, opencode_config: OpenCodeConfig | None = None
    ) -> None:
        """Initialize the code indexer with vector database configuration.

        Args:
            config: Vector database configuration including embedding model,
                storage settings, and chunking parameters.
            opencode_config: Optional OpenCode configuration for enhanced analysis.
        """
        self.config = config
        self.opencode_config = opencode_config
        self.vector_store = VectorStore(config)

        # Initialize OpenCode client for enhanced analysis if available
        self.opencode_client = None
        if opencode_config and opencode_config.enabled:
            from opencode_client import OpenCodeClient

            self.opencode_client = OpenCodeClient(opencode_config)

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

    def is_repository_indexed(self) -> bool:
        """Check if the repository has already been indexed.

        Returns:
            True if the repository appears to be already indexed, False otherwise.
        """
        return self.vector_store.is_indexed()

    def needs_reindexing(self, repo_path: str | Path) -> bool:
        """Check if the repository needs to be re-indexed.

        This method checks if there are source files that haven't been indexed
        or if existing files have been modified since indexing.

        Args:
            repo_path: Path to the repository to check.

        Returns:
            True if re-indexing is needed, False otherwise.
        """
        try:
            repo_path = Path(repo_path)
            if not self.vector_store.is_indexed():
                return True

            # Get currently indexed files
            indexed_files = self.vector_store.get_indexed_files()

            # Find current source files in repository
            current_files = self._find_code_files(repo_path, [], None)
            current_file_paths = {str(f.relative_to(repo_path)) for f in current_files}

            # Check if there are new files not in the index
            new_files = current_file_paths - indexed_files
            if new_files:
                logger.info(f"Found {len(new_files)} new files that need indexing")
                return True

            # Check if there are indexed files that no longer exist
            missing_files = indexed_files - current_file_paths
            if missing_files:
                logger.info(
                    f"Found {len(missing_files)} indexed files that no longer exist"
                )
                return True

            logger.info("Repository indexing is up to date")
            return False

        except Exception as e:
            logger.warning(f"Failed to check if reindexing is needed: {e}")
            return True  # Safe default - reindex if unsure

    async def index_repository(
        self,
        repo_path: str | Path,
        exclude_patterns: list[str] | None = None,
        target_files: list[str] | None = None,
    ) -> list[CodeContext]:
        """Index all relevant files in a repository with enhanced context grouping."""
        repo_path = Path(repo_path)
        exclude_patterns = exclude_patterns or []

        if not repo_path.exists():
            raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

        logger.info(f"Starting repository indexing: {repo_path}")
        logger.info(
            f"Indexing granularity: {'function-aware chunking' if self.config.chunk_size > 0 else 'file-level'}"
        )

        # Find all code files
        code_files = self._find_code_files(repo_path, exclude_patterns, target_files)
        logger.info(f"Found {len(code_files)} code files to index")

        # Process files and create contexts with OpenCode-enhanced relationships
        contexts = []
        for file_path in code_files:
            try:
                file_contexts = await self._process_file_with_opencode_context(
                    file_path
                )
                contexts.extend(file_contexts)
            except Exception as e:
                logger.warning(f"Failed to process file {file_path}: {e}")

        logger.info(f"Created {len(contexts)} code contexts")
        self._log_indexing_statistics(contexts, code_files)

        # Add contexts to vector store
        if contexts:
            self.vector_store.add_code_context(contexts)

        return contexts

    def search_relevant_context(
        self,
        problem_description: str,
        top_k: int = 10,  # TODO: We may need to increase this number
        file_filter: str | None = None,
    ) -> list[CodeContext]:
        """Search for code contexts relevant to a problem description."""
        results = self.vector_store.search_similar_contexts(
            query=problem_description, top_k=top_k, file_filter=file_filter
        )

        # Extract just the contexts (without scores)
        return [context for context, _ in results]

    def get_file_context(self, file_path: str) -> list[CodeContext]:
        """Get all contexts for a specific file."""
        return self.vector_store.get_context_by_file(file_path)

    async def search_relevant_context_with_dependencies(
        self,
        problem_description: str,
        top_k: int = 10,
        include_usage_analysis: bool = True,
        max_related_files: int = 20,
    ) -> list[CodeContext]:
        """Enhanced search that includes related code found through dependency analysis.

        This method extends the basic vector search by using OpenCode's file analysis
        capabilities to find code that uses or depends on the initially found relevant code.
        This helps capture a more complete picture of what might need to be considered
        when generating patches.

        Args:
            problem_description: Description of the problem to solve.
            top_k: Number of initial contexts to find via vector search.
            include_usage_analysis: Whether to find code that uses the target code.
            max_related_files: Maximum number of additional related files to include.

        Returns:
            List of CodeContext objects including both vector-search results and
            related code found through dependency analysis.
        """
        # Get base contexts from vector search
        base_contexts = self.search_relevant_context(
            problem_description=problem_description, top_k=top_k
        )

        if not self.opencode_client or not include_usage_analysis:
            return base_contexts

        logger.info(f"Analyzing dependencies for {len(base_contexts)} base contexts")

        # Find related code using OpenCode's file analysis
        related_contexts = await self._find_related_code_contexts(
            base_contexts=base_contexts, max_related_files=max_related_files
        )

        # Combine results, prioritizing base contexts
        all_contexts = base_contexts + related_contexts

        # Remove duplicates based on file path and content similarity
        unique_contexts = self._deduplicate_contexts(all_contexts)

        logger.info(
            f"Enhanced search found {len(base_contexts)} base + "
            f"{len(related_contexts)} related = {len(unique_contexts)} total contexts"
        )

        return unique_contexts[: top_k + max_related_files]

    async def _find_related_code_contexts(
        self, base_contexts: list[CodeContext], max_related_files: int = 20
    ) -> list[CodeContext]:
        """Find code contexts related to the base contexts using OpenCode file analysis."""
        related_contexts = []

        for context in base_contexts:
            try:
                # Extract function/class names from the context
                function_names = context.relevant_functions

                if not function_names:
                    continue

                # Use OpenCode to find usages of these functions
                for func_name in function_names[:5]:  # Limit to avoid too many requests
                    related_files = await self._find_function_usages(func_name)

                    for file_path in related_files[:3]:  # Limit files per function
                        if file_path != context.file_path:  # Skip same file
                            related_context = await self._create_context_from_file(
                                file_path
                            )
                            if related_context:
                                related_contexts.append(related_context)

                if len(related_contexts) >= max_related_files:
                    break

            except Exception as e:
                logger.warning(
                    f"Failed to find related contexts for {context.file_path}: {e}"
                )

        return related_contexts[:max_related_files]

    async def _find_function_usages(self, function_name: str) -> list[str]:
        """Use OpenCode's find API to locate files that use a specific function."""
        if not self.opencode_client:
            return []

        try:
            # Use OpenCode's find endpoint to search for function usage patterns
            search_patterns = [
                f"{function_name}(",  # Function calls
                f"from .* import.*{function_name}",  # Import statements
                f"import.*{function_name}",  # Direct imports
            ]

            found_files = set()
            for pattern in search_patterns:
                try:
                    # Use OpenCode's file search API
                    results = await self.opencode_client.find_in_files(pattern)

                    # Extract file paths from results
                    for result in results:
                        if isinstance(result, dict) and "path" in result:
                            found_files.add(result["path"])

                except Exception as e:
                    logger.debug(f"Search pattern '{pattern}' failed: {e}")

            return list(found_files)[:10]  # Limit results

        except Exception as e:
            logger.warning(f"Failed to find usages of function '{function_name}': {e}")
            return []

    async def _create_context_from_file(self, file_path: str) -> CodeContext | None:
        """Create a CodeContext from a file path, checking if it's already indexed."""
        try:
            # First check if we already have this file in our vector store
            existing_contexts = self.get_file_context(file_path)
            if existing_contexts:
                return existing_contexts[0]  # Return first context for this file

            # If not indexed, create a new context by reading the file
            full_path = (
                Path(
                    self.config.repository_path
                    if hasattr(self.config, "repository_path")
                    else "."
                )
                / file_path
            )

            if not full_path.exists():
                return None

            with open(full_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if not content.strip():
                return None

            # Determine language and create context
            language = self._detect_language(full_path)
            context = CodeContext(
                file_path=str(file_path),
                content=content,
                language=language,
                relevant_functions=self._extract_functions(content, language),
                dependencies=self._extract_dependencies(content, language),
            )

            return context

        except Exception as e:
            logger.warning(f"Failed to create context from file {file_path}: {e}")
            return None

    def _deduplicate_contexts(self, contexts: list[CodeContext]) -> list[CodeContext]:
        """Remove duplicate contexts based on file path."""
        seen_files = set()
        unique_contexts = []

        for context in contexts:
            if context.file_path not in seen_files:
                seen_files.add(context.file_path)
                unique_contexts.append(context)

        return unique_contexts

    async def _process_file_with_opencode_context(
        self, file_path: Path
    ) -> list[CodeContext]:
        """Process a file with OpenCode workspace analysis for enhanced context.

        Uses OpenCode's workspace APIs to find related files and symbols,
        providing richer context than simple directory-based grouping.
        """
        # Process the main file first
        contexts = self._process_file(file_path)

        if not self.opencode_client:
            # Fallback to basic processing if OpenCode not available
            return contexts

        try:
            # Enhance contexts with OpenCode workspace analysis
            for context in contexts:
                related_info = await self._find_opencode_related_files(
                    file_path, context
                )

                # Add the related information as metadata
                if not hasattr(context, "metadata"):
                    context.metadata = {}
                context.metadata.update(related_info)

        except Exception as e:
            logger.debug(f"OpenCode context enhancement failed for {file_path}: {e}")
            # Continue with basic contexts if OpenCode analysis fails

        return contexts

    async def _find_opencode_related_files(
        self, file_path: Path, context: CodeContext
    ) -> dict:
        """Use OpenCode to find files related to the given file and context."""
        related_info = {
            "related_files": [],
            "related_symbols": [],
            "workspace_analysis": False,
        }

        try:
            if not context.relevant_functions:
                return related_info

            # For each function in the context, find where it's used
            related_files = set()
            related_symbols = set()

            for func_name in context.relevant_functions[
                :3
            ]:  # Limit to avoid too many requests
                try:
                    # Find files that reference this function
                    usage_results = await self.opencode_client.find_in_files(
                        f"{func_name}("
                    )

                    for result in usage_results[:5]:  # Limit results
                        if isinstance(result, dict) and "path" in result:
                            result_path = result["path"]
                            if result_path != str(
                                file_path
                            ):  # Exclude the current file
                                related_files.add(result_path)

                    # Find symbol definitions related to this function
                    symbol_results = await self.opencode_client.find_symbols(func_name)

                    for symbol in symbol_results[:3]:  # Limit results
                        if isinstance(symbol, dict):
                            related_symbols.add(symbol.get("name", func_name))

                except Exception as e:
                    logger.debug(f"Failed to analyze function {func_name}: {e}")

            # Also look for files that import from this file
            if context.language == "python":
                module_name = file_path.stem  # filename without extension
                import_results = await self.opencode_client.find_in_files(
                    f"from.*{module_name}.*import|import.*{module_name}"
                )

                for result in import_results[:5]:
                    if isinstance(result, dict) and "path" in result:
                        result_path = result["path"]
                        if result_path != str(file_path):
                            related_files.add(result_path)

            related_info.update(
                {
                    "related_files": list(related_files)[
                        :10
                    ],  # Limit to 10 related files
                    "related_symbols": list(related_symbols)[
                        :10
                    ],  # Limit to 10 symbols
                    "workspace_analysis": True,
                }
            )

        except Exception as e:
            logger.debug(f"OpenCode workspace analysis failed: {e}")

        return related_info

    def _log_indexing_statistics(
        self, contexts: list[CodeContext], code_files: list[Path]
    ) -> None:
        """Log detailed statistics about the indexing process."""
        if not contexts:
            logger.warning("No contexts created during indexing")
            return

        # Calculate statistics
        total_files = len(code_files)
        total_contexts = len(contexts)
        contexts_per_file = total_contexts / total_files if total_files > 0 else 0

        # Language distribution
        language_counts = {}
        chunk_counts = {}
        opencode_enhanced = 0

        for context in contexts:
            lang = context.language
            language_counts[lang] = language_counts.get(lang, 0) + 1

            # Count chunks per file
            file_path = context.file_path
            chunk_counts[file_path] = chunk_counts.get(file_path, 0) + 1

            # Count OpenCode-enhanced contexts
            if hasattr(context, "metadata") and context.metadata.get(
                "workspace_analysis"
            ):
                opencode_enhanced += 1

        # Files with multiple chunks (indicating large files)
        multi_chunk_files = {
            path: count for path, count in chunk_counts.items() if count > 1
        }

        logger.info("Indexing Statistics:")
        logger.info(f"  - Total files: {total_files}")
        logger.info(f"  - Total contexts: {total_contexts}")
        logger.info(f"  - Average contexts per file: {contexts_per_file:.2f}")
        logger.info(
            f"  - Language distribution: {dict(sorted(language_counts.items()))}"
        )

        if self.opencode_client:
            logger.info(
                f"  - OpenCode-enhanced contexts: {opencode_enhanced}/{total_contexts}"
            )

        if multi_chunk_files:
            logger.info(
                f"  - Files split into multiple chunks: {len(multi_chunk_files)}"
            )
            logger.debug(
                f"  - Multi-chunk files: {dict(list(multi_chunk_files.items())[:5])}"
            )

        # Function extraction statistics
        total_functions = sum(len(context.relevant_functions) for context in contexts)
        contexts_with_functions = sum(
            1 for context in contexts if context.relevant_functions
        )

        if total_functions > 0:
            logger.info(f"  - Total functions extracted: {total_functions}")
            logger.info(
                f"  - Contexts with functions: {contexts_with_functions}/{total_contexts}"
            )

    def _find_code_files(
        self,
        repo_path: Path,
        exclude_patterns: list[str],
        target_files: list[str] | None = None,
    ) -> list[Path]:
        """Find all code files in the repository."""
        # TODO: Remove target_files, we should index the entire repository, not just specific files
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
        self, file_path: Path, repo_path: Path, exclude_patterns: list[str]
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
        # TODO: If required, we could use a library to detect the programming language in a more sophisticated way
        suffix = file_path.suffix.lower()

        for language, extensions in self.language_extensions.items():
            if suffix in extensions:
                return language

        return "unknown"

    def _chunk_file_content(
        self, file_path: Path, content: str, language: str
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

        # TODO: Improve this
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
