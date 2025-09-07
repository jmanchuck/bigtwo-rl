"""Random agent for fixed 1,365-action space.

This agent selects random actions from the legal action mask,
providing a baseline for evaluation and comparison.
"""

import numpy as np
from typing import Optional

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Random agent for fixed action space.

    This agent randomly selects from the legal actions provided by
    the action mask. It serves as a baseline for evaluation.
    """

    def __init__(self, name: str = "FixedRandom", seed: Optional[int] = None):
        """Initialize Fixed Action Random agent.

        Args:
            name: Agent name for identification
            seed: Random seed for reproducible behavior
        """
        super().__init__(name)
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def get_action(self, observation: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        """Get random action from legal actions.

        Args:
            observation: Game observation
            action_mask: 1365-dim boolean mask for legal actions

        Returns:
            Random legal action ID from 0-1364
        """
        if action_mask is not None:
            legal_actions = np.where(action_mask)[0]
            if len(legal_actions) > 0:
                return int(np.random.choice(legal_actions))
            else:
                # No legal actions (shouldn't happen in normal play)
                print("Warning: No legal actions available, selecting action 0")
                return 0

        # Fallback: uniform random from all 1365 actions
        return int(np.random.randint(0, 1365))

    def reset(self) -> None:
        """Reset agent state.

        For random agent, this resets the random seed if one was provided.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

    def set_seed(self, seed: int) -> None:
        """Set new random seed.

        Args:
            seed: New random seed
        """
        self.seed = seed
        np.random.seed(seed)

    def get_action_distribution(self, observation: np.ndarray, action_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Get uniform distribution over legal actions.

        Args:
            observation: Game observation
            action_mask: Legal action mask

        Returns:
            Probability distribution over actions
        """
        if action_mask is not None:
            # Uniform distribution over legal actions
            prob_dist = np.zeros(1365)
            legal_actions = np.where(action_mask)[0]
            if len(legal_actions) > 0:
                prob_dist[legal_actions] = 1.0 / len(legal_actions)
            return prob_dist
        else:
            # Uniform over all actions
            return np.ones(1365) / 1365.0


# Convenience function
def create_balanced_random_agent(name: str = "BalancedRandom") -> RandomAgent:
    """Create standard uniform random agent."""
    return RandomAgent(name=name)
