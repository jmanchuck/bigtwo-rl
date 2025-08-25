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
    
    def get_action(
        self, 
        observation: np.ndarray, 
        action_mask: Optional[np.ndarray] = None
    ) -> int:
        """Get random action from legal actions.
        
        Args:
            observation: Game observation (unused for random agent)
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
    
    def get_action_distribution(
        self, 
        observation: np.ndarray, 
        action_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Get uniform distribution over legal actions.
        
        Args:
            observation: Game observation (unused)
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


class WeightedRandomAgent(BaseAgent):
    """Random agent with action type preferences.
    
    This agent randomly selects actions but can weight certain
    types of actions (singles, pairs, etc.) higher than others.
    """
    
    def __init__(
        self, 
        name: str = "WeightedRandom",
        single_weight: float = 1.0,
        pair_weight: float = 0.8,
        triple_weight: float = 0.6,
        five_card_weight: float = 0.4,
        pass_weight: float = 0.2,
        seed: Optional[int] = None
    ):
        """Initialize Weighted Random agent.
        
        Args:
            name: Agent name
            single_weight: Weight for single card plays
            pair_weight: Weight for pair plays
            triple_weight: Weight for triple plays
            five_card_weight: Weight for five-card hands
            pass_weight: Weight for pass actions
            seed: Random seed
        """
        super().__init__(name)
        
        # Store weights for different action types
        self.weights = {
            'single': single_weight,
            'pair': pair_weight,
            'triple': triple_weight,
            'five_card': five_card_weight,
            'pass': pass_weight
        }
        
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
        # Import action space for action type identification
        from ..core.action_space import BigTwoActionSpace, HandType
        self.action_space = BigTwoActionSpace()
        self.hand_types = HandType
    
    def get_action(
        self, 
        observation: np.ndarray, 
        action_mask: Optional[np.ndarray] = None
    ) -> int:
        """Get weighted random action.
        
        Args:
            observation: Game observation (unused)
            action_mask: Legal action mask
            
        Returns:
            Weighted random action ID
        """
        if action_mask is None:
            # Fallback to uniform random
            return int(np.random.randint(0, 1365))
        
        legal_actions = np.where(action_mask)[0]
        if len(legal_actions) == 0:
            print("Warning: No legal actions available")
            return 0
        
        # Calculate weights for legal actions
        action_weights = []
        for action_id in legal_actions:
            action_spec = self.action_space.get_action_spec(action_id)
            hand_type = action_spec.hand_type
            
            if hand_type == self.hand_types.SINGLE:
                weight = self.weights['single']
            elif hand_type == self.hand_types.PAIR:
                weight = self.weights['pair']
            elif hand_type == self.hand_types.TRIPLE:
                weight = self.weights['triple']
            elif hand_type == self.hand_types.FIVE_CARD:
                weight = self.weights['five_card']
            elif hand_type == self.hand_types.PASS:
                weight = self.weights['pass']
            else:
                weight = 1.0  # Default weight
            
            action_weights.append(weight)
        
        # Normalize weights
        action_weights = np.array(action_weights)
        action_weights = action_weights / np.sum(action_weights)
        
        # Sample according to weights
        chosen_idx = np.random.choice(len(legal_actions), p=action_weights)
        return int(legal_actions[chosen_idx])
    
    def reset(self) -> None:
        """Reset agent state."""
        if self.seed is not None:
            np.random.seed(self.seed)
    
    def set_weights(self, **kwargs) -> None:
        """Update action type weights.
        
        Args:
            **kwargs: New weights for action types
        """
        for key, value in kwargs.items():
            if key in self.weights:
                self.weights[key] = value
            else:
                print(f"Warning: Unknown weight type '{key}'")


# Convenience functions for creating common random agent variants
def create_conservative_random_agent(name: str = "ConservativeRandom") -> WeightedRandomAgent:
    """Create random agent that prefers simpler moves."""
    return WeightedRandomAgent(
        name=name,
        single_weight=3.0,
        pair_weight=2.0,
        triple_weight=1.0,
        five_card_weight=0.5,
        pass_weight=1.5
    )


def create_aggressive_random_agent(name: str = "AggressiveRandom") -> WeightedRandomAgent:
    """Create random agent that prefers complex moves."""
    return WeightedRandomAgent(
        name=name,
        single_weight=0.5,
        pair_weight=1.0,
        triple_weight=2.0,
        five_card_weight=3.0,
        pass_weight=0.2
    )


def create_balanced_random_agent(name: str = "BalancedRandom") -> RandomAgent:
    """Create standard uniform random agent."""
    return RandomAgent(name=name)