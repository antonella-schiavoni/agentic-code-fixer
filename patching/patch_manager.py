"""Comprehensive patch management system for storing, tracking, and organizing code patches.

This module provides a centralized management system for patch candidates throughout
their lifecycle. It handles persistent storage, status tracking, metadata management,
and provides comprehensive querying capabilities for patch retrieval and analysis.

The patch manager ensures data integrity, provides detailed analytics, and serves as
the central repository for all patch-related operations within the automated code
fixing pipeline.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from core.types import PatchCandidate, PatchStatus

logger = logging.getLogger(__name__)


class PatchManager:
    """Centralized management system for patch candidates with persistent storage and tracking.

    This class provides comprehensive patch lifecycle management capabilities including
    storage, retrieval, status tracking, and metadata management. It maintains both
    consolidated and individual patch records for efficient access and detailed inspection.

    The manager supports various querying operations to filter patches by status, agent,
    target file, and other criteria. It also provides statistical analysis and export
    capabilities for monitoring patch generation and evaluation performance.

    Attributes:
        storage_dir: Directory path where patch data is persistently stored.
        patches_file: JSON file containing consolidated patch index.
        patches: In-memory dictionary mapping patch IDs to PatchCandidate objects.
    """

    def __init__(self, storage_dir: str | Path) -> None:
        """Initialize the patch manager with persistent storage configuration.

        Sets up the storage directory structure, initializes the patch index, and loads
        any existing patches from previous sessions. Creates necessary directories and
        files if they don't exist.

        Args:
            storage_dir: Directory path for storing patch data and metadata. Will be
                created if it doesn't exist.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.patches_file = self.storage_dir / "patches.json"
        self.patches: dict[str, PatchCandidate] = {}

        # Load existing patches if file exists
        self._load_patches()

        logger.info(f"Initialized patch manager with storage: {self.storage_dir}")

    def add_patch(self, patch: PatchCandidate) -> None:
        """Add a new patch candidate to the managed collection.

        Stores the patch in both the in-memory index and persistent storage, creating
        both a consolidated entry and an individual patch file for detailed inspection.
        Updates the main patches index automatically.

        Args:
            patch: The PatchCandidate object to add to the managed collection.
        """
        self.patches[patch.id] = patch
        self._save_patches()

        # Save individual patch file for easy inspection
        patch_file = self.storage_dir / f"patch_{patch.id}.json"
        with open(patch_file, "w", encoding="utf-8") as f:
            json.dump(patch.model_dump(), f, indent=2, default=str)

        logger.info(f"Added patch {patch.id} from agent {patch.agent_id}")

    def add_patches(self, patches: list[PatchCandidate]) -> None:
        """Add multiple patch candidates in a batch operation.

        Efficiently processes a collection of patches, adding each one to the managed
        storage with proper indexing and persistence. Provides bulk operation
        convenience for agent workflows that generate multiple patches.

        Args:
            patches: List of PatchCandidate objects to add to the collection.
        """
        for patch in patches:
            self.add_patch(patch)

        logger.info(f"Added {len(patches)} patches")

    def get_patch(self, patch_id: str) -> PatchCandidate | None:
        """Retrieve a specific patch candidate by its unique identifier.

        Args:
            patch_id: Unique identifier of the patch to retrieve.

        Returns:
            The PatchCandidate object if found, None otherwise.
        """
        return self.patches.get(patch_id)

    def get_patches_by_status(self, status: PatchStatus) -> list[PatchCandidate]:
        """Retrieve all patches that match a specific status.

        Filters the patch collection to return only those patches currently in the
        specified status state. Useful for finding patches ready for evaluation,
        testing, or application.

        Args:
            status: The PatchStatus to filter by (e.g., GENERATED, EVALUATED, APPLIED).

        Returns:
            List of PatchCandidate objects matching the specified status.
        """
        return [patch for patch in self.patches.values() if patch.status == status]

    def get_patches_by_agent(self, agent_id: str) -> list[PatchCandidate]:
        """Retrieve all patches generated by a specific AI agent.

        Filters patches by their originating agent, enabling analysis of agent
        performance and patch quality patterns. Useful for evaluating individual
        agent effectiveness and specialization outcomes.

        Args:
            agent_id: Unique identifier of the agent whose patches to retrieve.

        Returns:
            List of PatchCandidate objects generated by the specified agent.
        """
        return [patch for patch in self.patches.values() if patch.agent_id == agent_id]

    def get_patches_by_file(self, file_path: str) -> list[PatchCandidate]:
        """Retrieve all patches that target a specific source code file.

        Filters patches by their target file path, enabling file-specific analysis
        and comparison of different approaches to fixing issues in the same file.
        Useful for understanding patch concentration and conflict resolution.

        Args:
            file_path: Path of the target file to filter patches by.

        Returns:
            List of PatchCandidate objects targeting the specified file.
        """
        return [patch for patch in self.patches.values() if patch.file_path == file_path]

    def get_all_patches(self) -> list[PatchCandidate]:
        """Retrieve all patch candidates in the managed collection.

        Returns:
            Complete list of all PatchCandidate objects currently stored.
        """
        return list(self.patches.values())

    def update_patch_status(self, patch_id: str, status: PatchStatus) -> bool:
        """Update the lifecycle status of a specific patch.

        Changes the patch status to reflect its current state in the evaluation and
        application pipeline. Automatically persists the change to storage.

        Args:
            patch_id: Unique identifier of the patch to update.
            status: New PatchStatus to assign to the patch.

        Returns:
            True if the status was successfully updated, False if patch not found.
        """
        if patch_id not in self.patches:
            logger.warning(f"Patch {patch_id} not found")
            return False

        self.patches[patch_id].status = status
        self._save_patches()

        logger.info(f"Updated patch {patch_id} status to {status}")
        return True

    def update_patch_metadata(
        self,
        patch_id: str,
        metadata: dict[str, any]
    ) -> bool:
        """Update the metadata dictionary for a specific patch.

        Merges new metadata with existing patch metadata, allowing incremental
        updates without overwriting existing information. Commonly used to add
        evaluation results, test outcomes, or performance metrics.

        Args:
            patch_id: Unique identifier of the patch to update.
            metadata: Dictionary of metadata key-value pairs to merge.

        Returns:
            True if metadata was successfully updated, False if patch not found.
        """
        if patch_id not in self.patches:
            logger.warning(f"Patch {patch_id} not found")
            return False

        self.patches[patch_id].metadata.update(metadata)
        self._save_patches()

        logger.info(f"Updated patch {patch_id} metadata")
        return True

    def remove_patch(self, patch_id: str) -> bool:
        """Remove a patch completely from the managed collection and storage.

        Deletes the patch from both in-memory cache and persistent storage,
        including individual patch files. This operation is irreversible.

        Args:
            patch_id: Unique identifier of the patch to remove.

        Returns:
            True if the patch was successfully removed, False if not found.
        """
        if patch_id not in self.patches:
            logger.warning(f"Patch {patch_id} not found")
            return False

        del self.patches[patch_id]
        self._save_patches()

        # Remove individual patch file
        patch_file = self.storage_dir / f"patch_{patch_id}.json"
        if patch_file.exists():
            patch_file.unlink()

        logger.info(f"Removed patch {patch_id}")
        return True

    def get_patch_statistics(self) -> dict[str, any]:
        """Generate comprehensive statistics about the current patch collection.

        Analyzes the patch collection to provide detailed metrics including
        distribution by status, agent, target file, and confidence scores.
        Useful for monitoring system performance and agent effectiveness.

        Returns:
            Dictionary containing statistical breakdown of patches including:
            - total: Total number of patches
            - by_status: Count distribution by patch status
            - by_agent: Count distribution by generating agent
            - by_file: Count distribution by target file
            - average_confidence: Mean confidence score across all patches
        """
        if not self.patches:
            return {"total": 0}

        total = len(self.patches)
        by_status = {}
        by_agent = {}
        by_file = {}

        for patch in self.patches.values():
            # Count by status
            status = patch.status.value
            by_status[status] = by_status.get(status, 0) + 1

            # Count by agent
            agent = patch.agent_id
            by_agent[agent] = by_agent.get(agent, 0) + 1

            # Count by file
            file_path = patch.file_path
            by_file[file_path] = by_file.get(file_path, 0) + 1

        avg_confidence = sum(p.confidence_score for p in self.patches.values()) / total

        return {
            "total": total,
            "by_status": by_status,
            "by_agent": by_agent,
            "by_file": by_file,
            "average_confidence": avg_confidence,
        }

    def export_patches_summary(self, output_file: str | Path | None = None) -> dict[str, any]:
        """Export a comprehensive summary of all patches to a JSON file.

        Creates a detailed export containing both statistical analysis and individual
        patch summaries. The export includes key metrics for each patch without
        the full content, making it suitable for reporting and analysis.

        Args:
            output_file: Optional path for the export file. If not provided, saves
                to 'patches_summary.json' in the storage directory.

        Returns:
            Dictionary containing the complete summary data that was exported.
        """
        if output_file is None:
            output_file = self.storage_dir / "patches_summary.json"
        else:
            output_file = Path(output_file)

        summary = {
            "statistics": self.get_patch_statistics(),
            "patches": [
                {
                    "id": patch.id,
                    "description": patch.description,
                    "agent_id": patch.agent_id,
                    "file_path": patch.file_path,
                    "confidence_score": patch.confidence_score,
                    "status": patch.status.value,
                    "created_at": patch.created_at.isoformat(),
                    "line_range": f"{patch.line_start}-{patch.line_end}",
                }
                for patch in self.patches.values()
            ]
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Exported patch summary to {output_file}")
        return summary

    def clear_patches(self) -> None:
        """Remove all patches from both memory and persistent storage.

        Completely clears the patch collection, removing all data from both the
        in-memory cache and all persistent storage files. This operation is
        irreversible and should be used with caution.
        """
        self.patches.clear()
        self._save_patches()

        # Remove all individual patch files
        for patch_file in self.storage_dir.glob("patch_*.json"):
            patch_file.unlink()

        logger.info("Cleared all patches from storage")

    def _load_patches(self) -> None:
        """Load existing patches from persistent storage into memory.

        Reads the patches index file and reconstructs PatchCandidate objects
        from stored JSON data. Handles missing files gracefully and logs any
        errors encountered during the loading process.
        """
        if not self.patches_file.exists():
            return

        try:
            with open(self.patches_file, encoding="utf-8") as f:
                data = json.load(f)

            for patch_data in data:
                patch = PatchCandidate(**patch_data)
                self.patches[patch.id] = patch

            logger.info(f"Loaded {len(self.patches)} patches from storage")

        except Exception as e:
            logger.error(f"Failed to load patches from storage: {e}")

    def _save_patches(self) -> None:
        """Persist the current patch collection to storage.

        Serializes all patches to JSON format and writes them to the patches
        index file. Handles serialization errors gracefully and logs any
        issues encountered during the save operation.
        """
        try:
            patch_data = [patch.model_dump() for patch in self.patches.values()]

            with open(self.patches_file, "w", encoding="utf-8") as f:
                json.dump(patch_data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save patches to storage: {e}")

    def get_patches_for_evaluation(
        self,
        status: PatchStatus = PatchStatus.GENERATED,
        min_confidence: float = 0.0,
    ) -> list[PatchCandidate]:
        """Retrieve patches that are ready for evaluation based on status and confidence.

        Filters the patch collection to find candidates suitable for evaluation,
        applying both status and confidence threshold criteria. This helps ensure
        that only quality patches proceed to the evaluation stage.

        Args:
            status: Required patch status (defaults to GENERATED).
            min_confidence: Minimum confidence score threshold (defaults to 0.0).

        Returns:
            List of PatchCandidate objects meeting the evaluation criteria.
        """
        return [
            patch for patch in self.patches.values()
            if patch.status == status and patch.confidence_score >= min_confidence
        ]

    def mark_patch_evaluated(self, patch_id: str, evaluation_metadata: dict[str, any]) -> bool:
        """Mark a patch as evaluated and store the evaluation results.

        Updates the patch status to EVALUATED and stores detailed evaluation
        metadata including timestamps and evaluation outcomes. This creates
        a permanent record of the evaluation process for analysis and auditing.

        Args:
            patch_id: Unique identifier of the evaluated patch.
            evaluation_metadata: Dictionary containing evaluation results and metrics.

        Returns:
            True if the patch was successfully marked as evaluated, False if not found.
        """
        if patch_id not in self.patches:
            return False

        self.patches[patch_id].status = PatchStatus.EVALUATED
        self.patches[patch_id].metadata.update({
            "evaluation": evaluation_metadata,
            "evaluated_at": str(datetime.now())
        })
        self._save_patches()

        return True