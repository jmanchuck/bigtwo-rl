"""Greedy agent for fixed 1,365-action space Big Two."""

import numpy as np
from typing import Optional

from .base_agent import BaseAgent


class GreedyAgent(BaseAgent):
    """Greedy agent that always plays the lowest legal card/combination.

    This agent provides a consistent baseline by always selecting the first
    (lowest) legal action from the sorted action space.
    """

    def __init__(self, name: str = "Greedy"):
        """Initialize Greedy agent.

        Args:
            name: Agent name for identification
        """
        super().__init__(name)

    def get_action(self, observation: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        """Get greedy action - always select first legal action.

        The action space is organized so that lower action IDs correspond
        to lower-value plays, making this a consistent "play lowest" strategy.

        Args:
            observation: Game observation
            action_mask: 1365-dim boolean mask for legal actions

        Returns:
            Lowest legal action ID from 0-1364
        """
        if action_mask is not None:
            legal_actions = np.where(action_mask)[0]
            if len(legal_actions) > 0:
                # Return first (lowest) legal action
                return int(legal_actions[0])
            else:
                # No legal actions (shouldn't happen in normal play)
                print("Warning: No legal actions available, returning action 0")
                return 0

        # No action mask provided - default to action 0 (lowest single)
        return 0

    def reset(self) -> None:
        """Reset agent state.

        Greedy agent has no internal state to reset.
        """
        pass

    def get_action_distribution(self, observation: np.ndarray, action_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Get action distribution - deterministic greedy selection.

        Args:
            observation: Game observation
            action_mask: Legal action mask

        Returns:
            Probability distribution (1.0 for greedy action, 0.0 for others)
        """
        prob_dist = np.zeros(1365)

        if action_mask is not None:
            legal_actions = np.where(action_mask)[0]
            if len(legal_actions) > 0:
                # Put all probability on the first (lowest) legal action
                greedy_action = legal_actions[0]
                prob_dist[greedy_action] = 1.0
        else:
            # No mask - put probability on action 0
            prob_dist[0] = 1.0

        return prob_dist


# Convenience function
def create_greedy_agent(name: str = "Greedy") -> GreedyAgent:
    """Create a standard greedy agent."""
    return GreedyAgent(name)
