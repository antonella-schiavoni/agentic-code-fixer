"""LSP-powered patch validator for safe patch application.

This module provides validation logic that uses Language Server Protocol (LSP)
capabilities to verify that patches target the correct code constructs and
won't accidentally modify function definitions when they should only target
docstrings or function bodies.

The validator helps prevent the core issue where AI agents were incorrectly
including function signature lines in patches meant only for docstrings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from core.lsp_analyzer import CodeRange, LSPCodeAnalyzer, TargetType
from core.types import PatchCandidate
from opencode_client import OpenCodeClient

logger = logging.getLogger(__name__)


class ValidationResult(str, Enum):
    """Results of patch validation."""

    VALID = "valid"
    INVALID_RANGE = "invalid_range"
    WRONG_TARGET_TYPE = "wrong_target_type"
    OVERLAPPING_DEFINITIONS = "overlapping_definitions"
    LSP_ERROR = "lsp_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ValidationReport:
    """Detailed report of patch validation results."""

    result: ValidationResult
    patch_id: str
    message: str
    suggested_range: CodeRange | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] | None = None


class LSPPatchValidator:
    """LSP-powered validator for ensuring safe patch application.

    This validator uses OpenCode's LSP integration to perform semantic
    validation of patch candidates, ensuring they target only the intended
    code constructs and don't accidentally modify function definitions,
    class signatures, or other critical code elements.
    """

    def __init__(self, opencode_client: OpenCodeClient) -> None:
        """Initialize the LSP patch validator.

        Args:
            opencode_client: OpenCode client with LSP capabilities.
        """
        self.opencode_client = opencode_client
        self.lsp_analyzer = LSPCodeAnalyzer(opencode_client)

    async def validate_patch(
        self,
        session_id: str,
        patch: PatchCandidate,
        target_type: TargetType | None = None,
    ) -> ValidationReport:
        """Validate a patch candidate using LSP semantic analysis.

        Args:
            session_id: OpenCode session ID.
            patch: The patch candidate to validate.
            target_type: Expected target type (docstring, function_body, etc.).
                         If None, will be inferred from the patch content.

        Returns:
            ValidationReport with detailed validation results.
        """
        try:
            # Analyze the target file using LSP
            symbols = await self.lsp_analyzer.analyze_file(session_id, patch.file_path)

            if not symbols:
                return ValidationReport(
                    result=ValidationResult.LSP_ERROR,
                    patch_id=patch.id,
                    message=f"No LSP symbols found for file {patch.file_path}",
                    confidence=0.0,
                )

            # Find the symbol that the patch is targeting
            target_symbol = self._find_symbol_for_patch(symbols, patch)

            if not target_symbol:
                return ValidationReport(
                    result=ValidationResult.INVALID_RANGE,
                    patch_id=patch.id,
                    message=f"Patch targets lines {patch.line_start}-{patch.line_end} but no symbol found in that range",
                    confidence=0.8,
                )

            # Determine the intended target type if not provided
            if not target_type:
                target_type = self._infer_target_type(patch, target_symbol)

            # Validate the patch based on target type
            validation_result = self._validate_target_type(
                patch, target_symbol, target_type
            )

            if validation_result.result == ValidationResult.VALID:
                logger.info(f"Patch {patch.id} passed LSP validation")
            else:
                logger.warning(
                    f"Patch {patch.id} failed LSP validation: {validation_result.message}"
                )

            return validation_result

        except Exception as e:
            logger.error(f"LSP validation failed for patch {patch.id}: {e}")
            return ValidationReport(
                result=ValidationResult.UNKNOWN_ERROR,
                patch_id=patch.id,
                message=f"Validation error: {e!s}",
                confidence=0.0,
            )

    async def validate_patches(
        self, session_id: str, patches: list[PatchCandidate]
    ) -> list[ValidationReport]:
        """Validate multiple patches in batch.

        Args:
            session_id: OpenCode session ID.
            patches: List of patch candidates to validate.

        Returns:
            List of validation reports, one per patch.
        """
        reports = []
        for patch in patches:
            report = await self.validate_patch(session_id, patch)
            reports.append(report)

        valid_count = sum(
            1 for report in reports if report.result == ValidationResult.VALID
        )
        logger.info(
            f"Validated {len(patches)} patches: {valid_count} valid, {len(patches) - valid_count} invalid"
        )

        return reports

    async def suggest_corrected_patch(
        self, session_id: str, patch: PatchCandidate, target_type: TargetType
    ) -> PatchCandidate | None:
        """Suggest a corrected version of an invalid patch.

        Uses LSP analysis to find the correct line ranges for the intended
        target type and creates a corrected patch candidate.

        Args:
            session_id: OpenCode session ID.
            patch: The original patch candidate.
            target_type: The intended target type for the patch.

        Returns:
            Corrected patch candidate, or None if correction is not possible.
        """
        try:
            symbols = await self.lsp_analyzer.analyze_file(session_id, patch.file_path)

            if not symbols:
                return None

            # Find the symbol that should be targeted
            target_symbol = self._find_symbol_for_patch(symbols, patch)

            if not target_symbol:
                return None

            # Get the correct range for the target type
            correct_range = self._get_correct_range_for_target_type(
                target_symbol, target_type
            )

            if not correct_range:
                return None

            # Create corrected patch
            corrected_patch = PatchCandidate(
                content=patch.content,
                description=f"CORRECTED: {patch.description}",
                agent_id=f"{patch.agent_id}_lsp_corrected",
                file_path=patch.file_path,
                line_start=correct_range.start_line,
                line_end=correct_range.end_line,
                confidence_score=patch.confidence_score
                * 0.9,  # Slightly lower confidence
                metadata={
                    **patch.metadata,
                    "lsp_corrected": True,
                    "original_range": f"{patch.line_start}-{patch.line_end}",
                    "corrected_range": f"{correct_range.start_line}-{correct_range.end_line}",
                    "target_type": target_type.value,
                },
            )

            logger.info(
                f"Created corrected patch for {patch.id}: lines {patch.line_start}-{patch.line_end} → {correct_range.start_line}-{correct_range.end_line}"
            )
            return corrected_patch

        except Exception as e:
            logger.error(f"Failed to suggest correction for patch {patch.id}: {e}")
            return None

    def _find_symbol_for_patch(self, symbols: list, patch: PatchCandidate):
        """Find the symbol that a patch is targeting."""
        for symbol in symbols:
            # Check if patch range overlaps with symbol range
            if (
                patch.line_start >= symbol.range.start_line
                and patch.line_end <= symbol.range.end_line
            ):
                return symbol

            # Check children recursively
            if symbol.children:
                child_result = self._find_symbol_for_patch(symbol.children, patch)
                if child_result:
                    return child_result

        return None

    def _infer_target_type(self, patch: PatchCandidate, target_symbol) -> TargetType:
        """Infer the intended target type from patch content and context."""
        # Look for docstring-related content
        content_lower = patch.content.lower()
        description_lower = patch.description.lower()

        if any(
            keyword in content_lower or keyword in description_lower
            for keyword in [
                "docstring",
                "triple_quotes",
                "documentation",
                "description",
            ]
        ):
            return TargetType.DOCSTRING

        # Check if patch targets known docstring range
        if (
            target_symbol.docstring_range
            and patch.line_start >= target_symbol.docstring_range.start_line
            and patch.line_end <= target_symbol.docstring_range.end_line
        ):
            return TargetType.DOCSTRING

        # Check if patch targets known body range
        if (
            target_symbol.body_range
            and patch.line_start >= target_symbol.body_range.start_line
            and patch.line_end <= target_symbol.body_range.end_line
        ):
            return TargetType.FUNCTION_BODY

        # Default to entire function for functions, otherwise custom range
        if target_symbol.kind in ["function", "method"]:
            return TargetType.ENTIRE_FUNCTION
        else:
            return TargetType.CUSTOM_RANGE

    def _validate_target_type(
        self, patch: PatchCandidate, target_symbol, target_type: TargetType
    ) -> ValidationReport:
        """Validate that a patch correctly targets the intended construct type."""

        if target_type == TargetType.DOCSTRING:
            return self._validate_docstring_patch(patch, target_symbol)
        elif target_type == TargetType.FUNCTION_BODY:
            return self._validate_function_body_patch(patch, target_symbol)
        elif target_type == TargetType.ENTIRE_FUNCTION:
            return self._validate_entire_function_patch(patch, target_symbol)
        else:
            # For other types, just check if range is within symbol bounds
            if (
                patch.line_start >= target_symbol.range.start_line
                and patch.line_end <= target_symbol.range.end_line
            ):
                return ValidationReport(
                    result=ValidationResult.VALID,
                    patch_id=patch.id,
                    message=f"Patch correctly targets {target_type.value}",
                    confidence=0.8,
                )
            else:
                return ValidationReport(
                    result=ValidationResult.INVALID_RANGE,
                    patch_id=patch.id,
                    message=f"Patch range {patch.line_start}-{patch.line_end} exceeds symbol bounds",
                    confidence=0.9,
                )

    def _validate_docstring_patch(
        self, patch: PatchCandidate, target_symbol
    ) -> ValidationReport:
        """Validate a patch intended to modify only docstrings."""

        if not target_symbol.docstring_range:
            return ValidationReport(
                result=ValidationResult.WRONG_TARGET_TYPE,
                patch_id=patch.id,
                message=f"Function {target_symbol.name} has no docstring to modify",
                confidence=0.95,
            )

        # Check if patch exactly matches docstring range
        if (
            patch.line_start == target_symbol.docstring_range.start_line
            and patch.line_end == target_symbol.docstring_range.end_line
        ):
            return ValidationReport(
                result=ValidationResult.VALID,
                patch_id=patch.id,
                message="Patch correctly targets docstring lines only",
                confidence=0.95,
            )

        # Check if patch accidentally includes function definition
        if patch.line_start <= target_symbol.range.start_line:
            return ValidationReport(
                result=ValidationResult.OVERLAPPING_DEFINITIONS,
                patch_id=patch.id,
                message=f"Patch includes function definition line {target_symbol.range.start_line}",
                suggested_range=target_symbol.docstring_range,
                confidence=0.9,
            )

        # Check if patch is within docstring range but not exact
        if (
            patch.line_start >= target_symbol.docstring_range.start_line
            and patch.line_end <= target_symbol.docstring_range.end_line
        ):
            return ValidationReport(
                result=ValidationResult.VALID,
                patch_id=patch.id,
                message="Patch targets partial docstring range",
                confidence=0.8,
            )

        return ValidationReport(
            result=ValidationResult.INVALID_RANGE,
            patch_id=patch.id,
            message=f"Patch range {patch.line_start}-{patch.line_end} doesn't match docstring range {target_symbol.docstring_range.start_line}-{target_symbol.docstring_range.end_line}",
            suggested_range=target_symbol.docstring_range,
            confidence=0.85,
        )

    def _validate_function_body_patch(
        self, patch: PatchCandidate, target_symbol
    ) -> ValidationReport:
        """Validate a patch intended to modify function body only."""

        if not target_symbol.body_range:
            return ValidationReport(
                result=ValidationResult.WRONG_TARGET_TYPE,
                patch_id=patch.id,
                message=f"Function {target_symbol.name} has no identifiable body range",
                confidence=0.8,
            )

        # Check if patch is within body range
        if (
            patch.line_start >= target_symbol.body_range.start_line
            and patch.line_end <= target_symbol.body_range.end_line
        ):
            return ValidationReport(
                result=ValidationResult.VALID,
                patch_id=patch.id,
                message="Patch correctly targets function body",
                confidence=0.9,
            )

        # Check for overlaps with definition or docstring
        if patch.line_start <= target_symbol.range.start_line:
            return ValidationReport(
                result=ValidationResult.OVERLAPPING_DEFINITIONS,
                patch_id=patch.id,
                message=f"Patch includes function definition line {target_symbol.range.start_line}",
                suggested_range=target_symbol.body_range,
                confidence=0.9,
            )

        if (
            target_symbol.docstring_range
            and patch.line_start <= target_symbol.docstring_range.end_line
        ):
            return ValidationReport(
                result=ValidationResult.OVERLAPPING_DEFINITIONS,
                patch_id=patch.id,
                message="Patch overlaps with docstring lines",
                suggested_range=target_symbol.body_range,
                confidence=0.85,
            )

        return ValidationReport(
            result=ValidationResult.INVALID_RANGE,
            patch_id=patch.id,
            message=f"Patch range {patch.line_start}-{patch.line_end} doesn't match function body range",
            suggested_range=target_symbol.body_range,
            confidence=0.8,
        )

    def _validate_entire_function_patch(
        self, patch: PatchCandidate, target_symbol
    ) -> ValidationReport:
        """Validate a patch intended to modify an entire function."""

        # Check if patch matches the entire function range
        if (
            patch.line_start == target_symbol.range.start_line
            and patch.line_end == target_symbol.range.end_line
        ):
            return ValidationReport(
                result=ValidationResult.VALID,
                patch_id=patch.id,
                message="Patch correctly targets entire function",
                confidence=0.95,
            )

        # Check if patch is within function bounds
        if (
            patch.line_start >= target_symbol.range.start_line
            and patch.line_end <= target_symbol.range.end_line
        ):
            return ValidationReport(
                result=ValidationResult.VALID,
                patch_id=patch.id,
                message="Patch targets partial function range",
                confidence=0.8,
            )

        return ValidationReport(
            result=ValidationResult.INVALID_RANGE,
            patch_id=patch.id,
            message=f"Patch range {patch.line_start}-{patch.line_end} exceeds function bounds {target_symbol.range.start_line}-{target_symbol.range.end_line}",
            confidence=0.9,
        )

    def _get_correct_range_for_target_type(
        self, target_symbol, target_type: TargetType
    ) -> CodeRange | None:
        """Get the correct line range for a given target type."""

        if target_type == TargetType.DOCSTRING:
            return target_symbol.docstring_range
        elif target_type == TargetType.FUNCTION_BODY:
            return target_symbol.body_range
        elif target_type == TargetType.ENTIRE_FUNCTION:
            return target_symbol.range
        else:
            return target_symbol.range
