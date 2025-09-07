"""Strategic and complex reward functions for Big Two training."""

from typing import List, Optional
from .base_reward import BaseReward


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

    def move_bonus(self, move_cards: List[int]) -> float:
        # Reward for passing or complex moves
        if len(move_cards) == 0:
            return 0.05
        elif len(move_cards) == 2:
            return 0.2
        elif len(move_cards) == 3 or len(move_cards) == 5:
            return 0.5
        return 0.0

    def game_reward(
        self,
        winner_player: int,
        player_idx: int,
        cards_left: int,
        all_cards_left: Optional[List[int]] = None,
    ) -> float:
        """Advanced reward encouraging strategic play patterns."""
        if player_idx == winner_player:
            base_win = 1.0
            # Bonus for potentially strategic wins (check if others have many cards)
            if all_cards_left is not None:
                base_win += sum(all_cards_left) / 4.0  # Bonus for dominant wins
            return base_win

        # Strategic loss evaluation
        # 1. Position bonus: Reward good relative performance
        assert all_cards_left is not None
        sorted_indices = sorted(range(len(all_cards_left)), key=lambda i: all_cards_left[i])
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

        total_reward = position_reward + control_reward + efficiency_penalty

        return max(-2.0, total_reward)  # Cap minimum penalty

    def episode_bonus(self, games_won: int, total_games: int, avg_cards_left: float) -> float:
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

    def episode_bonus(self, games_won: int, total_games: int, avg_cards_left: float) -> float:
        """Small bonus for good episode performance (same as DefaultReward)."""
        win_rate = games_won / total_games if total_games > 0 else 0
        if win_rate > 0.6:
            return 0.5  # Bonus for >60% win rate
        return 0
