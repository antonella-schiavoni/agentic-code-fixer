"""Chess-style ELO ranking system for sophisticated patch candidate evaluation.

This module implements a robust ELO rating system adapted for code patch evaluation.
The ELO system, originally developed for chess rankings, provides an elegant solution
for ranking patches based on head-to-head comparison results without requiring
every patch to be compared against every other patch.

The system maintains dynamic ratings that evolve based on comparison outcomes,
with higher-rated patches having consistently outperformed others in evaluations.
This approach scales well to large numbers of patch candidates and provides
stable, meaningful rankings.
"""

from __future__ import annotations

import logging
import math

from core.types import EloRating, EvaluationResult, PatchCandidate

logger = logging.getLogger(__name__)


class EloRanker:
    """Chess-style ELO ranking system for dynamic patch candidate evaluation.

    This class implements the ELO rating algorithm to maintain dynamic rankings
    of patch candidates based on pairwise comparison results. The system provides
    a scalable approach to ranking that doesn't require exhaustive comparisons
    between all possible patch pairs.

    The ELO system works by:
    1. Starting all patches with the same initial rating
    2. Updating ratings based on comparison outcomes using expected vs actual results
    3. Converging to stable rankings that reflect relative patch quality

    Attributes:
        k_factor: Rating volatility parameter (higher = more volatile ratings).
        initial_rating: Starting rating for new patches (1200 is chess standard).
        ratings: Dictionary mapping patch IDs to their current ELO ratings.
    """

    def __init__(self, k_factor: int = 32, initial_rating: float = 1200.0) -> None:
        """Initialize the ELO ranking system with specified parameters.

        Args:
            k_factor: K-factor controlling rating change magnitude. Higher values
                make ratings more volatile and responsive to recent results.
            initial_rating: Starting rating assigned to new patches. Standard
                chess rating of 1200 provides a reasonable baseline.
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: dict[str, EloRating] = {}

        logger.info(f"Initialized ELO ranker (K={k_factor}, initial={initial_rating})")

    def initialize_patch_ratings(self, patches: list[PatchCandidate]) -> None:
        """Initialize ELO ratings for patch candidates."""
        for patch in patches:
            if patch.id not in self.ratings:
                self.ratings[patch.id] = EloRating(
                    patch_id=patch.id,
                    rating=self.initial_rating,
                )

        logger.info(f"Initialized ELO ratings for {len(patches)} patches")

    def update_ratings_from_evaluations(
        self,
        evaluation_results: list[EvaluationResult],
    ) -> None:
        """Update ELO ratings based on evaluation results."""
        if not evaluation_results:
            return

        for result in evaluation_results:
            self._update_rating_pair(
                patch_a_id=result.patch_a_id,
                patch_b_id=result.patch_b_id,
                winner_id=result.winner_id,
                confidence=result.confidence,
            )

        logger.info(f"Updated ELO ratings from {len(evaluation_results)} evaluations")

    def _update_rating_pair(
        self,
        patch_a_id: str,
        patch_b_id: str,
        winner_id: str,
        confidence: float,
    ) -> None:
        """Update ELO ratings for a pair of patches."""
        # Ensure ratings exist
        if patch_a_id not in self.ratings:
            self.ratings[patch_a_id] = EloRating(patch_id=patch_a_id, rating=self.initial_rating)
        if patch_b_id not in self.ratings:
            self.ratings[patch_b_id] = EloRating(patch_id=patch_b_id, rating=self.initial_rating)

        rating_a = self.ratings[patch_a_id]
        rating_b = self.ratings[patch_b_id]

        # Calculate expected scores
        expected_a = self._expected_score(rating_a.rating, rating_b.rating)
        expected_b = self._expected_score(rating_b.rating, rating_a.rating)

        # Determine actual scores based on winner and confidence
        if winner_id == patch_a_id:
            actual_a = confidence
            actual_b = 1.0 - confidence
        elif winner_id == patch_b_id:
            actual_a = 1.0 - confidence
            actual_b = confidence
        else:
            # Tie or unclear result
            actual_a = 0.5
            actual_b = 0.5

        # Update ratings using ELO formula
        new_rating_a = rating_a.rating + self.k_factor * (actual_a - expected_a)
        new_rating_b = rating_b.rating + self.k_factor * (actual_b - expected_b)

        # Update rating objects
        rating_a.rating = new_rating_a
        rating_a.matches_played += 1
        if actual_a > actual_b:
            rating_a.wins += 1
        elif actual_a < actual_b:
            rating_a.losses += 1

        rating_b.rating = new_rating_b
        rating_b.matches_played += 1
        if actual_b > actual_a:
            rating_b.wins += 1
        elif actual_b < actual_a:
            rating_b.losses += 1

        logger.debug(
            f"Updated ratings: {patch_a_id}: {rating_a.rating:.0f}, "
            f"{patch_b_id}: {rating_b.rating:.0f}"
        )

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))

    def get_ranked_patches(self, patches: list[PatchCandidate]) -> list[PatchCandidate]:
        """Get patches ranked by ELO rating."""
        # Sort patches by their ELO rating
        sorted_patches = sorted(
            patches,
            key=lambda p: self.ratings.get(p.id, EloRating(patch_id=p.id)).rating,
            reverse=True,
        )

        logger.info("Ranked patches by ELO rating")
        return sorted_patches

    def get_top_patch(self, patches: list[PatchCandidate]) -> PatchCandidate:
        """Get the highest-rated patch."""
        if not patches:
            raise ValueError("No patches provided")

        top_patch = max(
            patches,
            key=lambda p: self.ratings.get(p.id, EloRating(patch_id=p.id)).rating,
        )

        top_rating = self.ratings.get(top_patch.id, EloRating(patch_id=top_patch.id))
        logger.info(f"Top patch: {top_patch.id} (ELO: {top_rating.rating:.0f})")

        return top_patch

    def get_rating(self, patch_id: str) -> EloRating:
        """Get ELO rating for a specific patch."""
        return self.ratings.get(patch_id, EloRating(patch_id=patch_id, rating=self.initial_rating))

    def get_all_ratings(self) -> list[EloRating]:
        """Get all ELO ratings sorted by rating."""
        return sorted(self.ratings.values(), key=lambda r: r.rating, reverse=True)

    def tournament_selection(
        self,
        patches: list[PatchCandidate],
        tournament_size: int = 3,
    ) -> PatchCandidate:
        """Select a patch using tournament selection based on ELO ratings."""
        if len(patches) <= tournament_size:
            return self.get_top_patch(patches)

        import random

        # Select random subset for tournament
        tournament_patches = random.sample(patches, tournament_size)

        # Return highest rated patch from tournament
        return self.get_top_patch(tournament_patches)

    def simulate_tournament(
        self,
        patches: list[PatchCandidate],
        num_rounds: int = 100,
    ) -> list[PatchCandidate]:
        """Simulate a tournament to determine final rankings."""
        logger.info(f"Starting ELO tournament with {len(patches)} patches for {num_rounds} rounds")

        # Initialize ratings
        self.initialize_patch_ratings(patches)

        # Simulate random pairings for specified rounds
        import random

        for round_num in range(num_rounds):
            # Create random pairings
            shuffled_patches = patches.copy()
            random.shuffle(shuffled_patches)

            for i in range(0, len(shuffled_patches) - 1, 2):
                patch_a = shuffled_patches[i]
                patch_b = shuffled_patches[i + 1]

                # Simulate match result based on confidence scores
                prob_a_wins = patch_a.confidence_score / (
                    patch_a.confidence_score + patch_b.confidence_score
                )

                if random.random() < prob_a_wins:
                    winner_id = patch_a.id
                    confidence = prob_a_wins
                else:
                    winner_id = patch_b.id
                    confidence = 1.0 - prob_a_wins

                # Update ratings
                self._update_rating_pair(
                    patch_a_id=patch_a.id,
                    patch_b_id=patch_b.id,
                    winner_id=winner_id,
                    confidence=confidence,
                )

        # Return ranked patches
        final_rankings = self.get_ranked_patches(patches)
        logger.info(f"Tournament completed after {num_rounds} rounds")

        return final_rankings

    def get_tournament_stats(self) -> dict[str, any]:
        """Get statistics about the current tournament state."""
        if not self.ratings:
            return {"total_patches": 0}

        ratings_list = list(self.ratings.values())
        total_matches = sum(r.matches_played for r in ratings_list)
        avg_rating = sum(r.rating for r in ratings_list) / len(ratings_list)
        min_rating = min(r.rating for r in ratings_list)
        max_rating = max(r.rating for r in ratings_list)

        return {
            "total_patches": len(self.ratings),
            "total_matches": total_matches,
            "average_rating": avg_rating,
            "min_rating": min_rating,
            "max_rating": max_rating,
            "rating_spread": max_rating - min_rating,
            "k_factor": self.k_factor,
        }