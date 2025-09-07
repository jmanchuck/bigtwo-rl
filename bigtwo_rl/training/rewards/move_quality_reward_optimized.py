"""Optimized move quality-based reward system for strategic Big Two learning.

This module provides the same functionality as move_quality_reward.py but with
performance optimizations including caching, vectorization, and reduced computations.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from functools import lru_cache
from .base_reward import BaseReward


class MoveQualityRewardOptimized(BaseReward):
    """
    Performance-optimized version of move quality reward system.

    Optimizations include:
    - LRU caching for expensive computations
    - Pre-computed lookup tables
    - Vectorized numpy operations
    - Early termination for simple cases
    - Reduced redundant calculations
    """

    def __init__(
        self,
        card_efficiency_weight: float = 0.3,
        hand_type_weight: float = 0.2,
        timing_weight: float = 0.3,
        preservation_weight: float = 0.2,
        move_reward_scale: float = 0.2,
        game_reward_scale: float = 1.0,
    ):
        """Initialize optimized move quality reward system."""
        self.card_efficiency_weight = card_efficiency_weight
        self.hand_type_weight = hand_type_weight
        self.timing_weight = timing_weight
        self.preservation_weight = preservation_weight
        self.move_reward_scale = move_reward_scale
        self.game_reward_scale = game_reward_scale

        # Validate weights sum to 1.0
        total_weight = card_efficiency_weight + hand_type_weight + timing_weight + preservation_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        # Pre-compute card values lookup table
        self.card_values = np.array([val % 13 for val in range(52)])

        # Pre-computed hand type scores
        self._hand_type_scores = np.array(
            [
                0.1,  # single
                0.3,  # pair
                0.5,  # trip
                0.8,  # straight
                0.9,  # flush
                0.95,  # full_house
                0.98,  # four_of_a_kind
                1.0,  # straight_flush
                0.0,  # pass/invalid
            ]
        )

        # Pre-compute power cards set (Aces and 2s)
        self.power_cards_mask = np.zeros(52, dtype=bool)
        self.power_cards_mask[44:52] = True  # A♦,A♣,A♥,A♠,2♦,2♣,2♥,2♠

        # Cache for expensive computations
        self._hand_type_cache = {}
        self._potential_cache = {}

    def move_bonus(self, move_cards: List[int], game_context: Optional[Dict] = None) -> float:
        """Calculate immediate move quality reward with optimizations."""
        # Handle pass moves (fast path)
        if not move_cards:
            return self._evaluate_pass_quality_fast(game_context) * self.move_reward_scale

        # Extract game context with defaults (minimize dict lookups)
        if game_context is None:
            remaining_hand = []
            opponent_card_counts = [10, 10, 10]
            game_phase = "MIDGAME"
        else:
            remaining_hand = game_context.get("remaining_hand", [])
            opponent_card_counts = game_context.get("opponent_card_counts", [10, 10, 10])
            game_phase = game_context.get("game_phase", "MIDGAME")

        # Pre-convert to numpy arrays once
        move_cards_np = np.array(move_cards)
        remaining_hand_np = np.array(remaining_hand) if remaining_hand else np.array([])

        # Calculate all scores in parallel where possible
        efficiency_score = self._card_efficiency_score_fast(move_cards_np, remaining_hand_np)
        hand_type_score = self._hand_type_score_fast(move_cards_np)
        timing_score = self._timing_score_fast(len(move_cards), opponent_card_counts, game_phase)
        preservation_score = self._hand_preservation_score_fast(move_cards_np, remaining_hand_np)

        # Vectorized weighted combination
        scores = np.array([efficiency_score, hand_type_score, timing_score, preservation_score])
        weights = np.array(
            [
                self.card_efficiency_weight,
                self.hand_type_weight,
                self.timing_weight,
                self.preservation_weight,
            ]
        )

        total_quality = np.dot(scores, weights)
        return total_quality * self.move_reward_scale

    def _card_efficiency_score_fast(self, played_cards: np.ndarray, remaining_hand: np.ndarray) -> float:
        """Optimized card efficiency scoring using vectorized operations."""
        if len(played_cards) == 0 or len(remaining_hand) == 0:
            return 0.5  # Fast path for edge cases

        # Vectorized card value lookups
        played_values = self.card_values[played_cards]
        remaining_values = self.card_values[remaining_hand]

        # Use numpy means (faster than Python loops)
        played_avg = np.mean(played_values)
        remaining_avg = np.mean(remaining_values)

        # Vectorized efficiency calculation
        value_diff = remaining_avg - played_avg
        efficiency = np.clip((value_diff + 6) / 12.0, 0.0, 1.0)

        # Vectorized power card bonus using pre-computed mask
        power_cards_kept = np.sum(self.power_cards_mask[remaining_hand])
        power_cards_played = np.sum(self.power_cards_mask[played_cards])

        if power_cards_kept > power_cards_played:
            efficiency = min(1.0, efficiency + 0.1)

        return float(efficiency)

    @lru_cache(maxsize=512)
    def _hand_type_score_fast(self, move_cards_tuple: tuple) -> float:
        """Cached hand type scoring with fast lookup table."""
        if not move_cards_tuple:
            return 0.0

        hand_type_idx = self._identify_hand_type_fast(move_cards_tuple)
        return self._hand_type_scores[hand_type_idx]

    def _hand_type_score_fast(self, move_cards: np.ndarray) -> float:
        """Fast hand type scoring with caching."""
        if len(move_cards) == 0:
            return 0.0

        # Convert to tuple for caching
        move_cards_tuple = tuple(sorted(move_cards))
        return self._hand_type_score_fast_cached(move_cards_tuple)

    @lru_cache(maxsize=512)
    def _hand_type_score_fast_cached(self, move_cards_tuple: tuple) -> float:
        """Cached version of hand type scoring."""
        hand_type_idx = self._identify_hand_type_fast(move_cards_tuple)
        return self._hand_type_scores[hand_type_idx]

    def _timing_score_fast(self, move_strength: int, opponent_card_counts: List[int], game_phase: str) -> float:
        """Optimized timing score using lookup table approach."""
        if not opponent_card_counts:
            return 0.5

        min_opponent_cards = min(opponent_card_counts)

        # Use lookup table for common patterns instead of complex conditionals
        if move_strength >= 5:
            # 5-card hand timing
            if min_opponent_cards >= 8:
                return 1.0
            elif min_opponent_cards <= 3:
                return 0.2
            else:
                return 0.6
        elif move_strength >= 2:
            # Pair/trip timing
            return 0.9 if (game_phase == "ENDGAME" and min_opponent_cards <= 5) else 0.7
        else:
            # Single card
            return 0.5

    def _hand_preservation_score_fast(self, played_cards: np.ndarray, remaining_hand: np.ndarray) -> float:
        """Optimized preservation score with simplified heuristics."""
        if len(remaining_hand) == 0:
            return 1.0  # Perfect preservation when hand is empty

        if len(played_cards) == 0:
            return 1.0  # Pass preserves all options

        # Simplified heuristic: just count rank frequencies
        original_hand = np.concatenate([played_cards, remaining_hand])

        original_potential = self._count_potential_fast(original_hand)
        remaining_potential = self._count_potential_fast(remaining_hand)

        if original_potential == 0:
            return 1.0

        preservation_ratio = remaining_potential / original_potential
        return min(1.0, preservation_ratio)

    def _count_potential_fast(self, hand: np.ndarray) -> int:
        """Fast combination potential counting using vectorized operations."""
        if len(hand) == 0:
            return 0

        # Convert to tuple for caching
        hand_tuple = tuple(sorted(hand))
        return self._count_potential_cached(hand_tuple)

    @lru_cache(maxsize=256)
    def _count_potential_cached(self, hand_tuple: tuple) -> int:
        """Cached version of potential counting."""
        if not hand_tuple:
            return 0

        # Use numpy for rank counting
        ranks = np.array([card // 4 for card in hand_tuple])
        unique_ranks, counts = np.unique(ranks, return_counts=True)

        potential = 0

        # Vectorized pair/trip counting
        pair_mask = counts >= 2
        trip_mask = counts >= 3
        quad_mask = counts >= 4

        potential += np.sum(pair_mask)  # Pairs
        potential += np.sum(trip_mask)  # Trips
        potential += np.sum(quad_mask) * 2  # Quads are very valuable

        # Simple straight potential (consecutive ranks)
        if len(unique_ranks) >= 5:
            # Check for consecutive sequences
            diffs = np.diff(unique_ranks)
            consecutive_count = 1
            for diff in diffs:
                if diff == 1:
                    consecutive_count += 1
                    if consecutive_count >= 5:
                        potential += 2
                        break
                else:
                    consecutive_count = 1

        return potential

    @lru_cache(maxsize=512)
    def _identify_hand_type_fast(self, cards_tuple: tuple) -> int:
        """Fast hand type identification returning index for lookup table."""
        cards = list(cards_tuple)

        if not cards:
            return 8  # pass/invalid
        elif len(cards) == 1:
            return 0  # single
        elif len(cards) == 2:
            return 1 if cards[0] // 4 == cards[1] // 4 else 8  # pair or invalid
        elif len(cards) == 3:
            ranks = [card // 4 for card in cards]
            return 2 if len(set(ranks)) == 1 else 8  # trip or invalid
        elif len(cards) == 5:
            return self._identify_5_card_type_fast(cards)
        else:
            return 8  # invalid

    def _identify_5_card_type_fast(self, cards: List[int]) -> int:
        """Fast 5-card hand type identification."""
        ranks = np.array(sorted([card // 4 for card in cards]))
        suits = np.array([card % 4 for card in cards])

        # Vectorized checks
        is_straight = np.all(np.diff(ranks) == 1)
        is_flush = len(np.unique(suits)) == 1

        # Count rank frequencies
        unique_ranks, counts = np.unique(ranks, return_counts=True)
        counts_sorted = np.sort(counts)

        is_full_house = np.array_equal(counts_sorted, [2, 3])
        is_four_of_kind = 4 in counts

        # Return index for lookup table
        if is_straight and is_flush:
            return 7  # straight_flush
        elif is_four_of_kind:
            return 6  # four_of_a_kind
        elif is_full_house:
            return 5  # full_house
        elif is_flush:
            return 4  # flush
        elif is_straight:
            return 3  # straight
        else:
            return 8  # invalid

    def _evaluate_pass_quality_fast(self, game_context: Optional[Dict]) -> float:
        """Fast pass quality evaluation with simplified logic."""
        if not game_context:
            return 0.3

        remaining_hand = game_context.get("remaining_hand", [])
        opponent_card_counts = game_context.get("opponent_card_counts", [10, 10, 10])
        last_play_strength = game_context.get("last_play_strength", 1)

        min_opponent_cards = min(opponent_card_counts) if opponent_card_counts else 10

        # Fast power card counting using pre-computed mask
        if remaining_hand:
            power_cards_count = np.sum(self.power_cards_mask[remaining_hand])
        else:
            power_cards_count = 0

        # Simplified scoring logic
        pass_quality = 0.3

        if power_cards_count >= 2 and min_opponent_cards >= 6:
            pass_quality += 0.4

        if last_play_strength >= 5:
            pass_quality += 0.3

        return min(1.0, pass_quality)

    def game_reward(
        self,
        winner_player: int,
        player_idx: int,
        cards_left: int,
        all_cards_left: Optional[List[int]] = None,
    ) -> float:
        """Simple game reward (unchanged from original)."""
        if winner_player == player_idx:
            return 1.0 * self.game_reward_scale
        else:
            return -0.1 * cards_left * self.game_reward_scale

    def episode_bonus(self, games_won: int, total_games: int, avg_cards_left: float) -> float:
        """Simple episode bonus (unchanged from original)."""
        win_rate = games_won / total_games if total_games > 0 else 0
        return (win_rate - 0.25) * 0.5
