"""Vector store implementation using ChromaDB."""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from core.config import VectorDBConfig
from core.types import CodeContext

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector embeddings for code context using ChromaDB."""

    def __init__(self, config: VectorDBConfig) -> None:
        """Initialize vector store with configuration."""
        self.config = config
        self.embedding_model = SentenceTransformer(config.embedding_model)

        # Initialize ChromaDB client
        persist_dir = Path(config.persist_directory)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name, metadata={"hnsw:space": "cosine"}
        )

        logger.info(
            f"Initialized vector store with collection: {config.collection_name}"
        )

    def is_indexed(self) -> bool:
        """Check if the vector store contains any indexed data.

        Returns:
            True if the collection has indexed contexts, False otherwise.
        """
        try:
            count = self.collection.count()
            return count > 0
        except Exception as e:
            logger.warning(f"Failed to check if vector store is indexed: {e}")
            return False

    def get_indexed_files(self) -> set[str]:
        """Get the set of files that have been indexed.

        Returns:
            Set of file paths that are currently indexed in the vector store.
        """
        try:
            results = self.collection.get(include=["metadatas"])
            if results and results["metadatas"]:
                return {metadata["file_path"] for metadata in results["metadatas"]}
            return set()
        except Exception as e:
            logger.warning(f"Failed to get indexed files: {e}")
            return set()

    def add_code_context(self, contexts: list[CodeContext]) -> None:
        """Add code contexts to the vector store."""
        if not contexts:
            return

        # Generate embeddings for all contexts
        texts = [self._format_context_for_embedding(ctx) for ctx in contexts]
        embeddings = self.embedding_model.encode(texts).tolist()

        # Prepare data for ChromaDB
        ids = [f"{ctx.file_path}:{hash(ctx.content)}" for ctx in contexts]
        metadatas = [
            {
                "file_path": ctx.file_path,
                "language": ctx.language,
                "relevant_functions": ",".join(ctx.relevant_functions),
                "dependencies": ",".join(ctx.dependencies),
                "content_length": len(ctx.content),
            }
            for ctx in contexts
        ]

        # Update contexts with embeddings
        for ctx, embedding in zip(contexts, embeddings, strict=False):
            ctx.embedding = embedding

        # Add to collection
        self.collection.add(
            ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
        )

        logger.info(f"Added {len(contexts)} code contexts to vector store")

    def search_similar_contexts(
        self,
        query: str,
        top_k: int = 10,
        file_filter: str | None = None,
        language_filter: str | None = None,
        function_filter: str | None = None,
        dependency_filter: str | None = None,
        content_size_range: tuple[int, int] | None = None,
        languages: list[str] | None = None,
        file_patterns: list[str] | None = None,
    ) -> list[tuple[CodeContext, float]]:
        """Search for similar code contexts with advanced metadata filtering.

        Args:
            query: Text query for semantic similarity search.
            top_k: Maximum number of results to return.
            file_filter: Filter by file path substring.
            language_filter: Filter by specific programming language.
            function_filter: Filter by function name (searches in relevant_functions).
            dependency_filter: Filter by dependency/import (searches in dependencies).
            content_size_range: Filter by content size (min, max) in characters.
            languages: Filter by list of programming languages.
            file_patterns: Filter by list of file path patterns.

        Returns:
            List of (CodeContext, similarity_score) tuples.
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]

        # Build where clause for filtering
        where_conditions = []

        # Basic filters
        if file_filter:
            where_conditions.append({"file_path": {"$contains": file_filter}})
        if language_filter:
            where_conditions.append({"language": language_filter})
        if function_filter:
            where_conditions.append(
                {"relevant_functions": {"$contains": function_filter}}
            )
        if dependency_filter:
            where_conditions.append({"dependencies": {"$contains": dependency_filter}})

        # Content size range filtering
        if content_size_range:
            min_size, max_size = content_size_range
            where_conditions.append({"content_length": {"$gte": min_size}})
            where_conditions.append({"content_length": {"$lte": max_size}})

        # Multi-value filters
        if languages:
            where_conditions.append({"language": {"$in": languages}})
        if file_patterns:
            # Use OR logic for multiple file patterns
            pattern_conditions = [
                {"file_path": {"$contains": pattern}} for pattern in file_patterns
            ]
            where_conditions.append({"$or": pattern_conditions})

        # Combine all conditions with AND logic
        where_clause = {}
        if where_conditions:
            if len(where_conditions) == 1:
                where_clause = where_conditions[0]
            else:
                where_clause = {"$and": where_conditions}

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause if where_clause else None,
            include=["documents", "metadatas", "distances"],
        )

        # Convert results to CodeContext objects
        contexts_with_scores = []
        if results["documents"] and results["documents"][0]:
            for _i, (doc, metadata, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                    strict=False,
                )
            ):
                # Convert distance to similarity score (cosine distance -> similarity)
                similarity = 1.0 - distance

                context = CodeContext(
                    file_path=metadata["file_path"],
                    content=doc,
                    language=metadata["language"],
                    relevant_functions=(
                        metadata["relevant_functions"].split(",")
                        if metadata["relevant_functions"]
                        else []
                    ),
                    dependencies=(
                        metadata["dependencies"].split(",")
                        if metadata["dependencies"]
                        else []
                    ),
                )

                contexts_with_scores.append((context, similarity))

        logger.info(f"Found {len(contexts_with_scores)} similar contexts for query")
        return contexts_with_scores

    def get_context_by_file(self, file_path: str) -> list[CodeContext]:
        """Get all contexts for a specific file."""
        results = self.collection.query(
            query_embeddings=[
                [0.0] * self.embedding_model.get_sentence_embedding_dimension()
            ],
            n_results=1000,  # Large number to get all
            where={"file_path": file_path},
            include=["documents", "metadatas"],
        )

        contexts = []
        if results["documents"] and results["documents"][0]:
            for doc, metadata in zip(
                results["documents"][0], results["metadatas"][0], strict=False
            ):
                context = CodeContext(
                    file_path=metadata["file_path"],
                    content=doc,
                    language=metadata["language"],
                    relevant_functions=(
                        metadata["relevant_functions"].split(",")
                        if metadata["relevant_functions"]
                        else []
                    ),
                    dependencies=(
                        metadata["dependencies"].split(",")
                        if metadata["dependencies"]
                        else []
                    ),
                )
                contexts.append(context)

        return contexts

    def search_by_function(
        self, function_name: str, top_k: int = 10
    ) -> list[CodeContext]:
        """Search for contexts containing a specific function.

        Args:
            function_name: Name of the function to search for.
            top_k: Maximum number of results to return.

        Returns:
            List of CodeContext objects containing the function.
        """
        results = self.collection.get(
            where={"relevant_functions": {"$contains": function_name}},
            limit=top_k,
            include=["documents", "metadatas"],
        )

        contexts = []
        if results["documents"]:
            for doc, metadata in zip(
                results["documents"], results["metadatas"], strict=False
            ):
                context = CodeContext(
                    file_path=metadata["file_path"],
                    content=doc,
                    language=metadata["language"],
                    relevant_functions=(
                        metadata["relevant_functions"].split(",")
                        if metadata["relevant_functions"]
                        else []
                    ),
                    dependencies=(
                        metadata["dependencies"].split(",")
                        if metadata["dependencies"]
                        else []
                    ),
                )
                contexts.append(context)

        return contexts

    def search_by_dependency(
        self, dependency: str, top_k: int = 10
    ) -> list[CodeContext]:
        """Search for contexts that use a specific dependency/import.

        Args:
            dependency: Dependency/import to search for.
            top_k: Maximum number of results to return.

        Returns:
            List of CodeContext objects using the dependency.
        """
        results = self.collection.get(
            where={"dependencies": {"$contains": dependency}},
            limit=top_k,
            include=["documents", "metadatas"],
        )

        contexts = []
        if results["documents"]:
            for doc, metadata in zip(
                results["documents"], results["metadatas"], strict=False
            ):
                context = CodeContext(
                    file_path=metadata["file_path"],
                    content=doc,
                    language=metadata["language"],
                    relevant_functions=(
                        metadata["relevant_functions"].split(",")
                        if metadata["relevant_functions"]
                        else []
                    ),
                    dependencies=(
                        metadata["dependencies"].split(",")
                        if metadata["dependencies"]
                        else []
                    ),
                )
                contexts.append(context)

        return contexts

    def get_contexts_by_language(self, language: str) -> list[CodeContext]:
        """Get all contexts for a specific programming language.

        Args:
            language: Programming language to filter by.

        Returns:
            List of CodeContext objects for the specified language.
        """
        results = self.collection.get(
            where={"language": language},
            include=["documents", "metadatas"],
        )

        contexts = []
        if results["documents"]:
            for doc, metadata in zip(
                results["documents"], results["metadatas"], strict=False
            ):
                context = CodeContext(
                    file_path=metadata["file_path"],
                    content=doc,
                    language=metadata["language"],
                    relevant_functions=(
                        metadata["relevant_functions"].split(",")
                        if metadata["relevant_functions"]
                        else []
                    ),
                    dependencies=(
                        metadata["dependencies"].split(",")
                        if metadata["dependencies"]
                        else []
                    ),
                )
                contexts.append(context)

        return contexts

    def get_small_focused_contexts(self, max_size: int = 1000) -> list[CodeContext]:
        """Get contexts with small, focused code snippets.

        Args:
            max_size: Maximum content size in characters.

        Returns:
            List of small CodeContext objects.
        """
        results = self.collection.get(
            where={"content_length": {"$lte": max_size}},
            include=["documents", "metadatas"],
        )

        contexts = []
        if results["documents"]:
            for doc, metadata in zip(
                results["documents"], results["metadatas"], strict=False
            ):
                context = CodeContext(
                    file_path=metadata["file_path"],
                    content=doc,
                    language=metadata["language"],
                    relevant_functions=(
                        metadata["relevant_functions"].split(",")
                        if metadata["relevant_functions"]
                        else []
                    ),
                    dependencies=(
                        metadata["dependencies"].split(",")
                        if metadata["dependencies"]
                        else []
                    ),
                )
                contexts.append(context)

        return contexts

    def clear_collection(self) -> None:
        """Clear all data from the collection."""
        self.client.delete_collection(self.config.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name, metadata={"hnsw:space": "cosine"}
        )
        logger.info("Cleared vector store collection")

    def get_collection_stats(self) -> dict[str, int]:
        """Get statistics about the collection."""
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.config.collection_name,
        }

    def _format_context_for_embedding(self, context: CodeContext) -> str:
        """Format code context for embedding generation."""
        # Create a comprehensive text representation
        parts = [
            f"File: {context.file_path}",
            f"Language: {context.language}",
        ]

        if context.relevant_functions:
            parts.append(f"Functions: {', '.join(context.relevant_functions)}")

        if context.dependencies:
            parts.append(f"Dependencies: {', '.join(context.dependencies)}")

        parts.append("Content:")
        parts.append(context.content)

        return "\n".join(parts)
