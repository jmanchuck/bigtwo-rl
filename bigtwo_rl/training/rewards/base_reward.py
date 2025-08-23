"""Base reward class for Big Two training experiments."""

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