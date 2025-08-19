"""Base agent interface for Big Two agents."""

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """Base class for all Big Two agents."""

    def __init__(self, name: str):
        self.name = name
        self.wins = 0
        self.games_played = 0

    @abstractmethod
    def get_action(self, observation, action_mask=None):
        """
        Get action from agent given observation.

        Args:
            observation: Environment observation
            action_mask: Boolean mask of legal actions (optional)

        Returns:
            int: Action index
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset agent state for new game/episode."""
        pass

    def record_game_result(self, won: bool):
        """Record result of a game."""
        self.games_played += 1
        if won:
            self.wins += 1

    def get_win_rate(self) -> float:
        """Get current win rate."""
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played

    def reset_stats(self):
        """Reset win/loss statistics."""
        self.wins = 0
        self.games_played = 0
