"""
Balanced agent that uses move-type-first selection approach.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from .base_agent import BaseAgent


class BalancedRandomAgent(BaseAgent):
    """Random agent that balances move type selection like humans."""
    
    def __init__(self, name: str, move_type_weights: Optional[Dict[str, float]] = None):
        super().__init__(name)
        # Default weights favor simpler moves
        self.move_type_weights = move_type_weights or {
            "single": 0.4,
            "pair": 0.3,
            "trips": 0.15,
            "5-card": 0.15,
            "pass": 0.0  # Will be handled separately
        }
    
    def get_action(self, observation, action_mask=None):
        """Pick action by first selecting move type, then specific move."""
        if action_mask is None:
            return 0
        
        # Get legal action indices
        legal_actions = np.where(action_mask)[0]
        if len(legal_actions) == 0:
            return 0
        
        # We need access to the actual legal moves to categorize them
        # For now, fall back to uniform random - this needs environment context
        return random.choice(legal_actions)
    
    def get_action_with_moves(self, legal_moves: List[List[int]], action_mask=None, observation=None) -> int:
        """Enhanced action selection with access to legal moves."""
        if not legal_moves:
            return 0
        
        if action_mask is not None:
            legal_actions = np.where(action_mask)[0]
            valid_moves = [legal_moves[i] for i in legal_actions if i < len(legal_moves)]
        else:
            valid_moves = legal_moves
            legal_actions = list(range(len(legal_moves)))
        
        if not valid_moves:
            return 0
        
        # Categorize available moves by type
        move_categories = self._categorize_moves(valid_moves, legal_actions)
        
        # Select move type based on availability and weights
        selected_type = self._select_move_type(move_categories)
        
        # Select specific move within the chosen type
        if selected_type in move_categories:
            return random.choice(move_categories[selected_type])
        
        # Fallback to random selection
        return random.choice(legal_actions)
    
    def _categorize_moves(self, moves: List[List[int]], action_indices: List[int]) -> Dict[str, List[int]]:
        """Categorize moves by type and return action indices for each type."""
        categories = {
            "single": [],
            "pair": [],
            "trips": [],
            "5-card": [],
            "pass": []
        }
        
        for move, action_idx in zip(moves, action_indices):
            if len(move) == 0:
                categories["pass"].append(action_idx)
            elif len(move) == 1:
                categories["single"].append(action_idx)
            elif len(move) == 2:
                categories["pair"].append(action_idx)
            elif len(move) == 3:
                categories["trips"].append(action_idx)
            elif len(move) == 5:
                categories["5-card"].append(action_idx)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def _select_move_type(self, available_categories: Dict[str, List[int]]) -> str:
        """Select move type based on availability and weights."""
        if not available_categories:
            return "single"  # Fallback
        
        # Calculate weighted probabilities for available types only
        available_types = list(available_categories.keys())
        weights = [self.move_type_weights.get(move_type, 0.1) for move_type in available_types]
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(available_types)
        
        probabilities = [w / total_weight for w in weights]
        
        # Select based on weighted probability
        return np.random.choice(available_types, p=probabilities)
    
    def reset(self):
        """Reset agent state."""
        pass


class MoveTypeBalancedWrapper:
    """Wrapper that provides move-type-first selection for any agent."""
    
    def __init__(self, base_agent: BaseAgent, move_type_weights: Optional[Dict[str, float]] = None):
        self.base_agent = base_agent
        self.move_type_weights = move_type_weights or {
            "single": 0.35,
            "pair": 0.35, 
            "trips": 0.15,
            "5-card": 0.15,
            "pass": 0.0
        }
        self.name = f"Balanced({base_agent.name})"
    
    def get_action_with_moves(self, legal_moves: List[List[int]], action_mask=None, observation=None) -> int:
        """Select action using move-type-first approach."""
        if not legal_moves:
            return 0
        
        if action_mask is not None:
            legal_actions = np.where(action_mask)[0]
            valid_moves = [legal_moves[i] for i in legal_actions if i < len(legal_moves)]
            valid_indices = [i for i in legal_actions if i < len(legal_moves)]
        else:
            valid_moves = legal_moves
            valid_indices = list(range(len(legal_moves)))
        
        if not valid_moves:
            return 0
        
        # Categorize available moves
        move_categories = self._categorize_moves(valid_moves, valid_indices)
        
        # For pass moves, handle specially
        if "pass" in move_categories and len(move_categories) == 1:
            return move_categories["pass"][0]
        
        # Remove pass from consideration for now (add logic later if needed)
        play_categories = {k: v for k, v in move_categories.items() if k != "pass"}
        
        if not play_categories:
            return valid_indices[0]  # Fallback
        
        # Select move type
        selected_type = self._select_move_type(play_categories)
        
        if selected_type not in play_categories:
            return valid_indices[0]
        
        # For the selected type, let the base agent choose among options
        type_actions = play_categories[selected_type]
        
        if len(type_actions) == 1:
            return type_actions[0]
        
        # Create sub-mask for this move type
        sub_mask = np.zeros(len(legal_moves), dtype=bool)
        for action_idx in type_actions:
            if action_idx < len(legal_moves):
                sub_mask[action_idx] = True
        
        # Let base agent choose within this type
        if hasattr(self.base_agent, 'get_action') and observation is not None:
            action = self.base_agent.get_action(observation, sub_mask)
            if action in type_actions:
                return action
        
        # Fallback to random within type
        return random.choice(type_actions)
    
    def _categorize_moves(self, moves: List[List[int]], action_indices: List[int]) -> Dict[str, List[int]]:
        """Categorize moves by type."""
        categories = {"single": [], "pair": [], "trips": [], "5-card": [], "pass": []}
        
        for move, action_idx in zip(moves, action_indices):
            if len(move) == 0:
                categories["pass"].append(action_idx)
            elif len(move) == 1:
                categories["single"].append(action_idx)
            elif len(move) == 2:
                categories["pair"].append(action_idx)
            elif len(move) == 3:
                categories["trips"].append(action_idx)
            elif len(move) == 5:
                categories["5-card"].append(action_idx)
        
        return {k: v for k, v in categories.items() if v}
    
    def _select_move_type(self, available_categories: Dict[str, List[int]]) -> str:
        """Select move type based on availability and weights."""
        available_types = list(available_categories.keys())
        weights = [self.move_type_weights.get(move_type, 0.1) for move_type in available_types]
        
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(available_types)
        
        probabilities = [w / total_weight for w in weights]
        return np.random.choice(available_types, p=probabilities)
    
    def reset(self):
        """Reset both wrapper and base agent."""
        if hasattr(self.base_agent, 'reset'):
            self.base_agent.reset()
    
    def get_action(self, observation, action_mask=None):
        """Fallback method for compatibility."""
        return self.base_agent.get_action(observation, action_mask)