"""Vector store implementation using ChromaDB."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
import numpy as np
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
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"Initialized vector store with collection: {config.collection_name}")

    def add_code_context(self, contexts: List[CodeContext]) -> None:
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
        for ctx, embedding in zip(contexts, embeddings):
            ctx.embedding = embedding

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        logger.info(f"Added {len(contexts)} code contexts to vector store")

    def search_similar_contexts(
        self,
        query: str,
        top_k: int = 10,
        file_filter: Optional[str] = None,
        language_filter: Optional[str] = None,
    ) -> List[Tuple[CodeContext, float]]:
        """Search for similar code contexts."""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]

        # Build where clause for filtering
        where_clause = {}
        if file_filter:
            where_clause["file_path"] = {"$contains": file_filter}
        if language_filter:
            where_clause["language"] = language_filter

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause if where_clause else None,
            include=["documents", "metadatas", "distances"]
        )

        # Convert results to CodeContext objects
        contexts_with_scores = []
        if results["documents"] and results["documents"][0]:
            for i, (doc, metadata, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )
            ):
                # Convert distance to similarity score (cosine distance -> similarity)
                similarity = 1.0 - distance

                context = CodeContext(
                    file_path=metadata["file_path"],
                    content=doc,
                    language=metadata["language"],
                    relevant_functions=metadata["relevant_functions"].split(",") if metadata["relevant_functions"] else [],
                    dependencies=metadata["dependencies"].split(",") if metadata["dependencies"] else [],
                )

                contexts_with_scores.append((context, similarity))

        logger.info(f"Found {len(contexts_with_scores)} similar contexts for query")
        return contexts_with_scores

    def get_context_by_file(self, file_path: str) -> List[CodeContext]:
        """Get all contexts for a specific file."""
        results = self.collection.query(
            query_embeddings=[[0.0] * self.embedding_model.get_sentence_embedding_dimension()],
            n_results=1000,  # Large number to get all
            where={"file_path": file_path},
            include=["documents", "metadatas"]
        )

        contexts = []
        if results["documents"] and results["documents"][0]:
            for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
                context = CodeContext(
                    file_path=metadata["file_path"],
                    content=doc,
                    language=metadata["language"],
                    relevant_functions=metadata["relevant_functions"].split(",") if metadata["relevant_functions"] else [],
                    dependencies=metadata["dependencies"].split(",") if metadata["dependencies"] else [],
                )
                contexts.append(context)

        return contexts

    def clear_collection(self) -> None:
        """Clear all data from the collection."""
        self.client.delete_collection(self.config.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Cleared vector store collection")

    def get_collection_stats(self) -> Dict[str, int]:
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