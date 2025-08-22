"""Different reward functions for Big Two training experiments."""

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseReward(ABC):
    """Base class for custom reward functions with intermediate rewards."""

    @abstractmethod
    def game_reward(
        self,
        winner_player: int,
        player_idx: int,
        cards_left: int,
        all_cards_left: Optional[List[int]] = None,
    ) -> float:
        """
        Calculate immediate reward for a player after each game completion.

        Args:
            winner_player: Index of winning player
            player_idx: Index of player to calculate reward for
            cards_left: Number of cards left for this player
            all_cards_left: List of cards left for all players (for ranking-based rewards)

        Returns:
            float: Immediate reward value
        """
        pass

    @abstractmethod
    def episode_bonus(
        self, games_won: int, total_games: int, avg_cards_left: float
    ) -> float:
        """
        Calculate bonus reward at episode end based on overall performance.

        Args:
            games_won: Number of games won in this episode
            total_games: Total games played in this episode
            avg_cards_left: Average cards remaining when losing games

        Returns:
            float: Episode bonus reward
        """
        pass

    def move_bonus(self, move_cards: List[int]) -> float:
        """
        Calculate bonus reward for individual moves based on complexity.

        Args:
            move_cards: List of card indices that were played

        Returns:
            float: Move bonus reward (0.0 for most reward functions)
        """
        return 0.0


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


class StrategicReward(BaseReward):
    """Advanced reward structure to encourage sophisticated Big Two strategies.

    Encourages:
    - Strategic passing even when playable cards exist
    - Holding high-value cards (2s, Aces) for control
    - Saving small cards for 5-card combinations
    - Late-game positional advantage
    - Tempo control and timing
    """

    def __init__(self, control_bonus=0.3, position_bonus=0.2, efficiency_weight=0.4):
        """
        Args:
            control_bonus: Bonus for finishing with fewer cards when losing (strategic play)
            position_bonus: Bonus based on final ranking relative to card advantage
            efficiency_weight: Weight for card efficiency vs raw winning
        """
        self.control_bonus = control_bonus
        self.position_bonus = position_bonus
        self.efficiency_weight = efficiency_weight

    def game_reward(
        self,
        winner_player: int,
        player_idx: int,
        cards_left: int,
        all_cards_left: Optional[List[int]] = None,
    ) -> float:
        """Advanced reward encouraging strategic play patterns."""
        if player_idx == winner_player:
            # Winner reward - higher if won efficiently (fewer total moves implied)
            base_win = 2.0
            # Bonus for potentially strategic wins (check if others have many cards)
            if all_cards_left and max(all_cards_left) >= 8:
                base_win += 0.5  # Bonus for dominant wins
            return base_win

        if all_cards_left is None:
            # Fallback if ranking unavailable
            return max(-1.5, -0.15 * cards_left)

        # Strategic loss evaluation
        # 1. Position bonus: Reward good relative performance
        sorted_indices = sorted(
            range(len(all_cards_left)), key=lambda i: all_cards_left[i]
        )
        player_rank = sorted_indices.index(player_idx)  # 0=best, 3=worst
        position_reward = self.position_bonus * (3 - player_rank) / 3  # 0.2 to 0 range

        # 2. Control bonus: Reward finishing with few cards (strategic play)
        if cards_left <= 3:
            control_reward = self.control_bonus * (4 - cards_left) / 4  # Up to 0.3
        elif cards_left <= 6:
            control_reward = self.control_bonus * 0.5  # Moderate bonus
        else:
            control_reward = 0

        # 3. Efficiency penalty: Discourage being left with many cards
        efficiency_penalty = -self.efficiency_weight * min(cards_left / 13.0, 1.0)

        # 4. Strategic play bonus: Extra reward for 2nd place with very few cards
        strategic_bonus = 0
        if player_rank == 1 and cards_left <= 2:  # 2nd place, very few cards
            strategic_bonus = 0.4  # Suggests good strategic control

        total_reward = (
            position_reward + control_reward + efficiency_penalty + strategic_bonus
        )
        return max(-2.0, total_reward)  # Cap minimum penalty

    def episode_bonus(
        self, games_won: int, total_games: int, avg_cards_left: float
    ) -> float:
        """Bonus for consistent strategic performance across episode."""
        win_rate = games_won / total_games if total_games > 0 else 0

        # Strategic consistency bonus
        if win_rate >= 0.6 and avg_cards_left <= 4.0:
            return 1.0  # Excellent strategic play
        elif win_rate >= 0.4 and avg_cards_left <= 5.0:
            return 0.6  # Good strategic play
        elif avg_cards_left <= 3.0:  # Good card management even if fewer wins
            return 0.4
        elif win_rate >= 0.5:  # Decent win rate
            return 0.2

        return 0


class ComplexMoveReward(BaseReward):
    """Reward function that encourages playing complex card combinations.

    Provides a flat bonus for 5-card hands (straights, flushes, full houses, etc.)
    while maintaining standard win/loss reward structure.
    """

    def __init__(
        self,
        five_card_bonus: float = 0.0,
        pair_bonus: float = 0.0,
        base_reward_scale: float = 1.0,
    ):
        """
        Args:
            five_card_bonus: Bonus reward for each 5-card hand played (default: 0.1)
            base_reward_scale: Scaling factor for base win/loss rewards (default: 1.0)
        """
        self.five_card_bonus = five_card_bonus
        self.pair_bonus = pair_bonus
        self.base_reward_scale = base_reward_scale

    def move_bonus(self, move_cards: List[int]) -> float:
        """Provide bonus for complex moves (5-card hands)."""
        if len(move_cards) == 5:
            return self.five_card_bonus
        elif len(move_cards) == 2:
            return self.pair_bonus
        return 0.0

    def game_reward(
        self,
        winner_player: int,
        player_idx: int,
        cards_left: int,
        all_cards_left: Optional[List[int]] = None,
    ) -> float:
        """Standard win/loss reward structure similar to DefaultReward."""
        if player_idx == winner_player:
            return self.base_reward_scale * 1.0  # Winner gets positive reward
        else:
            # Non-winners: penalty based on cards remaining (same as DefaultReward)
            if cards_left >= 10:
                penalty = -1.0  # Large penalty for many cards
            elif cards_left >= 7:
                penalty = cards_left * -0.1
            elif cards_left >= 3:
                penalty = cards_left * -0.05
            elif cards_left >= 1:
                penalty = cards_left * -0.02
            else:
                penalty = 0
            return self.base_reward_scale * penalty

    def episode_bonus(
        self, games_won: int, total_games: int, avg_cards_left: float
    ) -> float:
        """Small bonus for good episode performance (same as DefaultReward)."""
        win_rate = games_won / total_games if total_games > 0 else 0
        if win_rate > 0.6:
            return self.base_reward_scale * 0.5  # Bonus for >60% win rate
        return 0
