"""Greedy agent that plays lowest legal card."""

import numpy as np
from typing import Optional, Any
from .base_agent import BaseAgent


class GreedyAgent(BaseAgent):
    """Greedy baseline agent - always plays lowest legal card, otherwise passes."""

    def __init__(self, name: str = "Greedy") -> None:
        super().__init__(name)
        self.env_ref = None  # Will be set by tournament runner

    def set_env_reference(self, env: Any) -> None:
        """Set reference to environment for accessing legal moves."""
        self.env_ref = env

    def get_action(
        self, observation: np.ndarray, action_mask: Optional[np.ndarray] = None
    ) -> int:
        """Play lowest legal card (single card preferred), otherwise pass."""
        if self.env_ref is None:
            # Fallback to random if no env reference
            if action_mask is not None:
                legal_actions = np.where(action_mask)[0]
                if len(legal_actions) > 0:
                    return legal_actions[0]
            return 0

        # Get legal moves from environment
        legal_moves = self.env_ref.env.legal_moves(self.env_ref.env.current_player)

        # Find lowest single card first
        for i, move in enumerate(legal_moves):
            if len(move) == 1:  # Single card
                return i

        # If no singles, try first legal move
        if legal_moves:
            return 0
        return 0  # Should not happen

    def reset(self) -> None:
        """Nothing to reset for greedy agent."""
        pass
