"""Simple and sparse reward functions for Big Two training."""

from typing import List, Optional
from .base_reward import BaseReward


class DefaultReward(BaseReward):
    """Default reward structure with immediate game rewards."""

    def game_reward(
        self,
        winner_player: int,
        player_idx: int,
        cards_left: int,
        all_cards_left: Optional[List[int]] = None,
    ) -> float:
        """Immediate reward after each game."""
        if player_idx == winner_player:
            return 1.0  # Winner gets positive reward
        else:
            # Non-winners: penalty based on cards remaining
            if cards_left >= 10:
                return -1.0  # Large penalty for many cards
            elif cards_left >= 7:
                return cards_left * -0.1
            elif cards_left >= 3:
                return cards_left * -0.05
            elif cards_left >= 1:
                return cards_left * -0.02
            else:
                return 0

    def episode_bonus(
        self, games_won: int, total_games: int, avg_cards_left: float
    ) -> float:
        """Small bonus for good episode performance."""
        win_rate = games_won / total_games if total_games > 0 else 0
        if win_rate > 0.6:
            return 0.5  # Bonus for >60% win rate
        return 0


class SparseReward(BaseReward):
    """Sparse reward - only win/loss, no card count penalty."""

    def game_reward(
        self,
        winner_player: int,
        player_idx: int,
        cards_left: int,
        all_cards_left: Optional[List[int]] = None,
    ) -> float:
        """Simple win/loss reward."""
        if player_idx == winner_player:
            return 1.0  # Win
        else:
            return -0.25  # Loss (reduced from -1 for better learning)

    def episode_bonus(
        self, games_won: int, total_games: int, avg_cards_left: float
    ) -> float:
        """No episode bonus for sparse rewards."""
        return 0


class AggressivePenaltyReward(BaseReward):
    """Higher penalties for losing with many cards."""

    def game_reward(
        self,
        winner_player: int,
        player_idx: int,
        cards_left: int,
        all_cards_left: Optional[List[int]] = None,
    ) -> float:
        """Aggressive penalties for poor performance."""
        if player_idx == winner_player:
            return 2.0  # Higher win reward
        else:
            # Steeper penalties (scaled down for immediate rewards)
            if cards_left >= 8:
                return cards_left * -0.5
            elif cards_left >= 4:
                return cards_left * -0.3
            else:
                return cards_left * -0.1

    def episode_bonus(
        self, games_won: int, total_games: int, avg_cards_left: float
    ) -> float:
        """Large bonus for avoiding penalties."""
        win_rate = games_won / total_games if total_games > 0 else 0
        if win_rate > 0.7:
            return 2.0  # Large bonus for high win rate
        elif avg_cards_left < 5.0:  # Good at minimizing cards when losing
            return 0.5
        return 0


class ProgressiveReward(BaseReward):
    """Reward for making progress (fewer cards = better reward)."""

    def game_reward(
        self,
        winner_player: int,
        player_idx: int,
        cards_left: int,
        all_cards_left: Optional[List[int]] = None,
    ) -> float:
        """Reward progress in reducing cards."""
        if player_idx == winner_player:
            return 1.5
        else:
            # Only reward when close to winning (1-2 cards), otherwise scale penalty
            if cards_left <= 2:
                return 0.5 - (cards_left * 0.2)  # 2 cards = 0.1, 1 card = 0.3
            else:
                # Scale penalty from -0.2 (3 cards) to -1.0 (13 cards)
                return -0.2 - ((cards_left - 3) * 0.08)  # Linear scaling

    def episode_bonus(
        self, games_won: int, total_games: int, avg_cards_left: float
    ) -> float:
        """Bonus for consistent progress."""
        if avg_cards_left < 2.0:  # Very good at minimizing cards
            return 1.0
        elif avg_cards_left < 5.0:  # Decent progress
            return 0.3
        return 0


class RankingReward(BaseReward):
    """Reward based on final ranking among all players."""

    def game_reward(self, winner_player, player_idx, cards_left, all_cards_left):
        """Reward based on ranking."""
        if player_idx == winner_player:
            return 1.0

        # Rank players by cards left (fewer = better rank)
        sorted_players = sorted(enumerate(all_cards_left), key=lambda x: x[1])
        rank = next(i for i, (p, _) in enumerate(sorted_players) if p == player_idx)

        # Rank 0 = winner (handled above), 1 = 2nd place, etc.
        return 0.5 - rank * 0.2  # 2nd place gets 0.3, 3rd gets 0.1, last gets -0.1

    def episode_bonus(
        self, games_won: int, total_games: int, avg_cards_left: float
    ) -> float:
        """Bonus for consistent ranking performance."""
        win_rate = games_won / total_games if total_games > 0 else 0
        # Bonus based on both wins and low card count when losing
        combined_score = win_rate + (1.0 - min(avg_cards_left / 10.0, 1.0))
        return max(0, combined_score - 1.0)


class ScoreMarginReward(BaseReward):
    """Reward based on score margin / normalized card advantage.

    Provides a continuous signal reflecting how well the player did relative to
    opponents. Output is roughly bounded in [-1, 1].
    """

    def game_reward(
        self,
        winner_player: Optional[int],
        player_idx: int,
        cards_left: int,
        all_cards_left: Optional[List[int]],
    ) -> float:
        # If provided a winner, give a clear positive/negative signal
        if winner_player is not None:
            if player_idx == winner_player:
                base = 1.0
            else:
                base = -1.0
        else:
            base = 0.0

        # margin: average opponents' remaining cards minus player's remaining cards
        if all_cards_left is None or len(all_cards_left) <= 1:
            margin = 0.0
        else:
            opponents = [c for i, c in enumerate(all_cards_left) if i != player_idx]
            avg_opp = sum(opponents) / len(opponents)
            # normalize by max cards (13)
            margin = (avg_opp - cards_left) / 13.0

        # Combine base win/loss signal with margin, weighted to keep magnitude reasonable
        return float(0.5 * base + 0.5 * margin)

    def episode_bonus(
        self, games_won: int, total_games: int, avg_cards_left: float
    ) -> float:
        # Encourage consistent performance: normalized win-rate minus avg_cards_left factor
        win_rate = games_won / total_games if total_games > 0 else 0.0
        normalized_cards = 1.0 - min(avg_cards_left / 13.0, 1.0)
        return float(0.5 * win_rate + 0.5 * normalized_cards - 0.5)
