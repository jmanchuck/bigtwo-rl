"""Random agent that picks random legal actions."""

import random
import numpy as np
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """Random baseline agent - picks random legal action."""
    
    def __init__(self, name="Random"):
        super().__init__(name)
    
    def get_action(self, observation, action_mask=None):
        """Pick random legal action."""
        if action_mask is not None:
            legal_actions = np.where(action_mask)[0]
            if len(legal_actions) > 0:
                return random.choice(legal_actions)
        return 0  # Fallback
    
    def reset(self):
        """Nothing to reset for random agent."""
        pass