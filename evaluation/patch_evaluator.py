"""Sophisticated patch evaluation system using Claude for intelligent comparison.

This module implements a comprehensive evaluation system that uses Claude's
reasoning capabilities to compare patch candidates and determine which solutions
are superior. It supports both pairwise AB testing and tournament-style evaluation
methodologies.

The evaluator goes beyond simple metrics by providing detailed reasoning about
why one patch is better than another, considering factors like correctness,
maintainability, security, and performance implications.
"""

from __future__ import annotations

import asyncio
import logging

from core.config import EvaluationConfig, OpenCodeConfig
from core.types import EvaluationResult, PatchCandidate
from opencode_client import OpenCodeClient

logger = logging.getLogger(__name__)


class PatchEvaluator:
    """Advanced patch evaluation system using OpenCode's LLM provider management.

    This class provides sophisticated evaluation capabilities for comparing patch
    candidates using LLM reasoning through OpenCode SST. It can perform pairwise
    comparisons, tournament-style evaluations, and detailed analysis of code quality
    factors while leveraging OpenCode's provider authentication system.

    The evaluator considers multiple dimensions when comparing patches:
    - Correctness and bug fixing effectiveness
    - Code maintainability and readability
    - Security implications and best practices
    - Performance considerations
    - Edge case handling and robustness

    Attributes:
        config: Evaluation configuration parameters and model settings.
        opencode_config: OpenCode configuration for LLM provider management.
        opencode_client: OpenCode client for LLM communication.
        session_id: Active OpenCode session for evaluation.
    """

    def __init__(
        self,
        config: EvaluationConfig,
        opencode_config: OpenCodeConfig
    ) -> None:
        """Initialize the patch evaluator with OpenCode-powered comparison capabilities.

        Args:
            config: Evaluation configuration including model parameters, comparison
                methodology, and quality thresholds.
            opencode_config: OpenCode configuration for LLM provider management.
        """
        self.config = config
        self.opencode_config = opencode_config
        self.opencode_client = OpenCodeClient(opencode_config) if opencode_config.enabled else None
        self.session_id: str | None = None

        logger.info("Initialized patch evaluator with OpenCode SST")

    async def evaluate_patches_pairwise(
        self,
        patches: list[PatchCandidate],
        problem_description: str,
        original_code: str,
    ) -> list[EvaluationResult]:
        """Perform comprehensive pairwise evaluation of patch candidates using AB testing.

        This method conducts systematic pairwise comparisons between all patch
        candidates using Claude's advanced reasoning capabilities. Each comparison
        provides detailed analysis of why one patch is superior to another.

        The evaluation process generates all possible unique pairs and performs
        concurrent evaluations to efficiently determine relative patch quality.
        Results include confidence scores and detailed reasoning for each comparison.

        Args:
            patches: List of patch candidates to evaluate and compare.
            problem_description: Original problem that patches are attempting to solve.
            original_code: Baseline code before any patches are applied.

        Returns:
            List of EvaluationResult objects containing pairwise comparison outcomes,
            including winner identification, confidence scores, and detailed reasoning.

        Raises:
            ValueError: If fewer than 2 patches provided for comparison.
            Exception: If Claude API communication fails during evaluation.
        """
        if len(patches) < 2:
            logger.warning("Need at least 2 patches for pairwise evaluation")
            return []

        logger.info(f"Starting pairwise evaluation of {len(patches)} patches")

        # Generate all unique pairs
        pairs = []
        for i in range(len(patches)):
            for j in range(i + 1, len(patches)):
                pairs.append((patches[i], patches[j]))

        logger.info(f"Generated {len(pairs)} patch pairs for evaluation")

        # Evaluate pairs concurrently
        tasks = [
            self._evaluate_patch_pair(
                patch_a=pair[0],
                patch_b=pair[1],
                problem_description=problem_description,
                original_code=original_code,
            )
            for pair in pairs
        ]

        # Execute with concurrency control
        semaphore = asyncio.Semaphore(3)  # Limit concurrent evaluations

        async def bounded_evaluation(task):
            async with semaphore:
                return await task

        try:
            results = await asyncio.gather(
                *[bounded_evaluation(task) for task in tasks],
                return_exceptions=True
            )

            # Filter successful results
            evaluation_results = []
            for result in results:
                if isinstance(result, EvaluationResult):
                    evaluation_results.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Evaluation failed: {result}")

            logger.info(f"Completed {len(evaluation_results)} pairwise evaluations")
            return evaluation_results

        except Exception as e:
            logger.error(f"Pairwise evaluation failed: {e}")
            return []

    async def _evaluate_patch_pair(
        self,
        patch_a: PatchCandidate,
        patch_b: PatchCandidate,
        problem_description: str,
        original_code: str,
    ) -> EvaluationResult:
        """Evaluate a pair of patches and determine the winner."""
        if not self.opencode_client or not self.session_id:
            logger.error("No active OpenCode session for patch evaluation")
            return EvaluationResult(
                patch_a_id=patch_a.id,
                patch_b_id=patch_b.id,
                winner_id=patch_a.id,
                confidence=0.5,
                reasoning="No OpenCode session available for evaluation",
            )

        try:
            system_prompt = self._create_evaluation_system_prompt()
            user_prompt = self._create_evaluation_user_prompt(
                patch_a=patch_a,
                patch_b=patch_b,
                problem_description=problem_description,
                original_code=original_code,
            )

            # Use OpenCode's LLM provider management
            response = await self.opencode_client.send_prompt(
                session_id=self.session_id,
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                agent_id="patch_evaluator"
            )

            # Parse evaluation response
            evaluation_result = self._parse_opencode_evaluation_response(
                response=response,
                patch_a=patch_a,
                patch_b=patch_b,
            )

            logger.debug(f"Evaluated patches {patch_a.id} vs {patch_b.id}")
            return evaluation_result

        except Exception as e:
            logger.error(f"Failed to evaluate patch pair: {e}")
            # Return default result favoring patch A
            return EvaluationResult(
                patch_a_id=patch_a.id,
                patch_b_id=patch_b.id,
                winner_id=patch_a.id,
                confidence=0.5,
                reasoning="Evaluation failed, defaulting to first patch",
            )

    def _create_evaluation_system_prompt(self) -> str:
        """Create system prompt for patch evaluation."""
        return """
You are an expert code reviewer tasked with comparing two different patches that attempt to fix the same problem.

Your job is to:
1. Analyze both patches for correctness, maintainability, and effectiveness
2. Consider security implications, performance impact, and code quality
3. Determine which patch better solves the stated problem
4. Provide a confidence score for your decision

Evaluation criteria:
- Correctness: Does the patch actually fix the problem?
- Safety: Does it introduce new bugs or security issues?
- Maintainability: Is the code clean and readable?
- Performance: Does it have acceptable performance characteristics?
- Completeness: Does it handle edge cases appropriately?

Respond with a JSON object containing:
- winner: "patch_a" or "patch_b"
- confidence: float between 0.0 and 1.0
- reasoning: detailed explanation of your decision

Example:
```json
{
  "winner": "patch_a",
  "confidence": 0.85,
  "reasoning": "Patch A provides a more robust solution by handling edge cases that Patch B ignores. While both fix the immediate issue, Patch A's error handling makes it more suitable for production use."
}
```
        """.strip()

    def _create_evaluation_user_prompt(
        self,
        patch_a: PatchCandidate,
        patch_b: PatchCandidate,
        problem_description: str,
        original_code: str,
    ) -> str:
        """Create user prompt for patch evaluation."""
        return f"""
Problem Description:
{problem_description}

Original Code:
```
{original_code}
```

Patch A (from {patch_a.agent_id}):
Description: {patch_a.description}
Confidence: {patch_a.confidence_score}
Lines {patch_a.line_start}-{patch_a.line_end}:
```
{patch_a.content}
```

Patch B (from {patch_b.agent_id}):
Description: {patch_b.description}
Confidence: {patch_b.confidence_score}
Lines {patch_b.line_start}-{patch_b.line_end}:
```
{patch_b.content}
```

Please evaluate these two patches and determine which one better solves the problem. Consider all aspects of code quality, correctness, and maintainability.
        """.strip()

    def _parse_opencode_evaluation_response(
        self,
        response: dict[str, any],
        patch_a: PatchCandidate,
        patch_b: PatchCandidate,
    ) -> EvaluationResult:
        """Parse the evaluation response from OpenCode's LLM."""
        try:
            # Extract content from OpenCode response
            content = response.get("content") or response.get("text") or response.get("response", "")

            # Extract JSON from response
            import json
            import re

            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                eval_data = json.loads(json_match.group(1))
            else:
                # Try to find JSON object in response
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
                if json_match:
                    eval_data = json.loads(json_match.group(0))
                else:
                    raise ValueError("No valid JSON found in evaluation response")

            # Determine winner
            winner_key = eval_data.get("winner", "").lower()
            if winner_key == "patch_a":
                winner_id = patch_a.id
            elif winner_key == "patch_b":
                winner_id = patch_b.id
            else:
                # Default to higher confidence patch
                winner_id = patch_a.id if patch_a.confidence_score >= patch_b.confidence_score else patch_b.id

            return EvaluationResult(
                patch_a_id=patch_a.id,
                patch_b_id=patch_b.id,
                winner_id=winner_id,
                confidence=float(eval_data.get("confidence", 0.5)),
                reasoning=eval_data.get("reasoning", "No reasoning provided"),
                metadata={
                    "evaluator_model": self.config.model_name,
                    "opencode_session": self.session_id,
                    "raw_response": content,
                    "opencode_response": response,
                },
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse OpenCode evaluation response: {e}")
            # Return default result
            return EvaluationResult(
                patch_a_id=patch_a.id,
                patch_b_id=patch_b.id,
                winner_id=patch_a.id,  # Default to patch A
                confidence=0.5,
                reasoning=f"Failed to parse evaluation response: {e}",
            )

    def _parse_evaluation_response(
        self,
        response: any,  # Kept for backward compatibility
        patch_a: PatchCandidate,
        patch_b: PatchCandidate,
    ) -> EvaluationResult:
        """Legacy method for backward compatibility."""
        logger.warning("Using deprecated _parse_evaluation_response, use _parse_opencode_evaluation_response instead")
        return EvaluationResult(
            patch_a_id=patch_a.id,
            patch_b_id=patch_b.id,
            winner_id=patch_a.id,
            confidence=0.5,
            reasoning="Legacy evaluation method used",
        )

    def set_session_id(self, session_id: str) -> None:
        """Set the OpenCode session ID for evaluation.

        Args:
            session_id: OpenCode session ID to use for evaluation.
        """
        self.session_id = session_id
        logger.debug(f"PatchEvaluator assigned to session {session_id}")

    async def find_best_patch(
        self,
        patches: list[PatchCandidate],
        problem_description: str,
        original_code: str,
        min_comparisons: int | None = None,
    ) -> PatchCandidate | None:
        """Find the best patch using tournament-style evaluation."""
        if not patches:
            return None

        if len(patches) == 1:
            return patches[0]

        min_comparisons = min_comparisons or self.config.min_comparisons_per_patch

        logger.info(f"Finding best patch among {len(patches)} candidates")

        # Get pairwise evaluation results
        evaluation_results = await self.evaluate_patches_pairwise(
            patches=patches,
            problem_description=problem_description,
            original_code=original_code,
        )

        if not evaluation_results:
            logger.warning("No evaluation results available, returning highest confidence patch")
            return max(patches, key=lambda p: p.confidence_score)

        # Count wins for each patch
        win_counts = {patch.id: 0 for patch in patches}
        total_comparisons = {patch.id: 0 for patch in patches}

        for result in evaluation_results:
            # Count participation
            total_comparisons[result.patch_a_id] += 1
            total_comparisons[result.patch_b_id] += 1

            # Count wins (weighted by confidence)
            if result.winner_id == result.patch_a_id:
                win_counts[result.patch_a_id] += result.confidence
            elif result.winner_id == result.patch_b_id:
                win_counts[result.patch_b_id] += result.confidence

        # Calculate win rates
        win_rates = {}
        for patch_id in win_counts:
            if total_comparisons[patch_id] > 0:
                win_rates[patch_id] = win_counts[patch_id] / total_comparisons[patch_id]
            else:
                win_rates[patch_id] = 0.0

        # Find patch with highest win rate
        best_patch_id = max(win_rates, key=win_rates.get)
        best_patch = next(p for p in patches if p.id == best_patch_id)

        logger.info(
            f"Best patch: {best_patch_id} "
            f"(win rate: {win_rates[best_patch_id]:.2f}, "
            f"comparisons: {total_comparisons[best_patch_id]})"
        )

        return best_patch

    def get_evaluation_stats(self, evaluation_results: list[EvaluationResult]) -> dict[str, any]:
        """Get statistics about evaluation results."""
        if not evaluation_results:
            return {"total_evaluations": 0}

        total = len(evaluation_results)
        avg_confidence = sum(r.confidence for r in evaluation_results) / total

        # Count wins by patch
        patch_wins = {}
        for result in evaluation_results:
            winner = result.winner_id
            patch_wins[winner] = patch_wins.get(winner, 0) + 1

        return {
            "total_evaluations": total,
            "average_confidence": avg_confidence,
            "patch_wins": patch_wins,
            "evaluator_model": self.config.model_name,
        }