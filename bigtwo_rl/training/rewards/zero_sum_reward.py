"""Zero-sum reward function matching the successful Big Two implementation."""

from typing import List, Optional
from .base_reward import BaseReward


class ZeroSumReward(BaseReward):
    """Pure zero-sum reward structure matching the successful implementation.
    
    Winner gets the sum of all other players' remaining cards.
    Losers get negative penalty equal to their remaining cards.
    This creates a pure zero-sum game where winner's reward exactly equals
    the sum of all penalties.
    
    Key features:
    - Simple, clear signal for the model
    - Winner reward = sum of all other penalties  
    - No complex bonuses or strategic considerations
    - Matches the reward structure of the high-performing reference implementation
    """

    def __init__(self, normalization_factor: float = 5.0):
        """
        Args:
            normalization_factor: Divide rewards by this (reference uses 5.0)
                                 Results in rewards roughly in range [-2.6, +6.0]
        """
        self.normalization_factor = normalization_factor

    def game_reward(
        self,
        winner_player: int,
        player_idx: int,
        cards_left: int,
        all_cards_left: Optional[List[int]] = None,
    ) -> float:
        """Pure zero-sum reward structure.
        
        Args:
            winner_player: Index of winning player (0-3)
            player_idx: Index of current player (0-3) 
            cards_left: Number of cards this player has left
            all_cards_left: List of cards left for all players [p0, p1, p2, p3]
        
        Returns:
            Normalized reward value
        """
        if all_cards_left is None:
            # Fallback if we don't have full game state
            if player_idx == winner_player:
                # Estimate winner reward (assume others have ~8 cards average)
                estimated_total = 24  # 3 other players * 8 cards
                return estimated_total / self.normalization_factor
            else:
                return -cards_left / self.normalization_factor
        
        if player_idx == winner_player:
            # Winner gets sum of all other players' remaining cards
            total_other_cards = sum(all_cards_left[i] for i in range(4) if i != winner_player)
            return total_other_cards / self.normalization_factor
        else:
            # Losers get negative penalty equal to their remaining cards
            return -cards_left / self.normalization_factor

    def episode_bonus(
        self, games_won: int, total_games: int, avg_cards_left: float
    ) -> float:
        """No episode bonus - keep it simple like the reference implementation."""
        return 0.0

    def move_bonus(self, move_cards: List[int]) -> float:
        """No move bonuses - pure zero-sum structure."""
        return 0.0