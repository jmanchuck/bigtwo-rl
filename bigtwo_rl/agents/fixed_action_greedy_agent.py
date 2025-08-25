"""Greedy agent for fixed 1,365-action space.

This agent implements various greedy strategies for Big Two,
providing stronger baselines than random agents for evaluation.
"""

import numpy as np
from typing import Optional, List, Dict, Any

from .base_agent import BaseAgent
from ..core.action_space import BigTwoActionSpace, HandType, ActionSpec


class FixedActionGreedyAgent(BaseAgent):
    """Greedy agent for fixed action space.
    
    This agent implements a simple greedy strategy that prefers
    to play the lowest/weakest cards first to get rid of them.
    """
    
    def __init__(self, name: str = "FixedGreedy", strategy: str = "lowest_first"):
        """Initialize Fixed Action Greedy agent.
        
        Args:
            name: Agent name for identification
            strategy: Greedy strategy to use ("lowest_first", "highest_first", "singles_first")
        """
        super().__init__(name)
        self.strategy = strategy
        self.action_space = BigTwoActionSpace()
        
        # Strategy mappings
        self.strategy_functions = {
            "lowest_first": self._lowest_first_strategy,
            "highest_first": self._highest_first_strategy,
            "singles_first": self._singles_first_strategy,
            "clear_hand": self._clear_hand_strategy,
            "adaptive": self._adaptive_strategy,
        }
        
        if strategy not in self.strategy_functions:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.strategy_functions.keys())}")
    
    def get_action(
        self, 
        observation: np.ndarray, 
        action_mask: Optional[np.ndarray] = None
    ) -> int:
        """Get greedy action based on strategy.
        
        Args:
            observation: Game observation (unused by basic greedy)
            action_mask: 1365-dim boolean mask for legal actions
            
        Returns:
            Selected action ID from greedy strategy
        """
        if action_mask is None:
            # No mask provided - return first action (shouldn't happen)
            return 0
        
        legal_actions = np.where(action_mask)[0]
        if len(legal_actions) == 0:
            print("Warning: No legal actions available")
            return 0
        
        # Apply selected strategy
        strategy_func = self.strategy_functions[self.strategy]
        return strategy_func(legal_actions, observation)
    
    def _lowest_first_strategy(self, legal_actions: np.ndarray, observation: np.ndarray) -> int:
        """Strategy: Play the lowest/weakest cards first.
        
        Priority: Singles < Pairs < Triples < Five-cards, then by card indices
        """
        best_action = None
        best_score = float('inf')
        
        for action_id in legal_actions:
            action_spec = self.action_space.get_action_spec(action_id)
            
            # Score based on hand type and card indices
            if action_spec.hand_type == HandType.PASS:
                # Avoid passing unless necessary
                score = 10000
            elif action_spec.hand_type == HandType.SINGLE:
                # Prefer singles with lowest indices
                score = 100 + min(action_spec.card_indices) if action_spec.card_indices else 100
            elif action_spec.hand_type == HandType.PAIR:
                # Pairs are okay, but prefer lower cards
                score = 200 + min(action_spec.card_indices) if action_spec.card_indices else 200
            elif action_spec.hand_type == HandType.TRIPLE:
                # Triples are less preferred
                score = 300 + min(action_spec.card_indices) if action_spec.card_indices else 300
            elif action_spec.hand_type == HandType.FIVE_CARD:
                # Five-card hands are complex, use with caution
                score = 500 + min(action_spec.card_indices) if action_spec.card_indices else 500
            else:
                score = 1000
            
            if score < best_score:
                best_score = score
                best_action = action_id
        
        return int(best_action) if best_action is not None else int(legal_actions[0])
    
    def _highest_first_strategy(self, legal_actions: np.ndarray, observation: np.ndarray) -> int:
        """Strategy: Play the highest/strongest cards first.
        
        This is generally a poor strategy but useful for comparison.
        """
        best_action = None
        best_score = -1
        
        for action_id in legal_actions:
            action_spec = self.action_space.get_action_spec(action_id)
            
            # Score based on hand type and card indices (higher is better)
            if action_spec.hand_type == HandType.PASS:
                score = -1000  # Avoid passing
            elif action_spec.hand_type == HandType.FIVE_CARD:
                # Prefer complex hands
                score = 500 + max(action_spec.card_indices) if action_spec.card_indices else 500
            elif action_spec.hand_type == HandType.TRIPLE:
                score = 300 + max(action_spec.card_indices) if action_spec.card_indices else 300
            elif action_spec.hand_type == HandType.PAIR:
                score = 200 + max(action_spec.card_indices) if action_spec.card_indices else 200
            elif action_spec.hand_type == HandType.SINGLE:
                score = 100 + max(action_spec.card_indices) if action_spec.card_indices else 100
            else:
                score = 0
            
            if score > best_score:
                best_score = score
                best_action = action_id
        
        return int(best_action) if best_action is not None else int(legal_actions[0])
    
    def _singles_first_strategy(self, legal_actions: np.ndarray, observation: np.ndarray) -> int:
        """Strategy: Strongly prefer singles to clear hand quickly."""
        # First, try to find singles
        for action_id in legal_actions:
            action_spec = self.action_space.get_action_spec(action_id)
            if action_spec.hand_type == HandType.SINGLE:
                return int(action_id)
        
        # If no singles available, fall back to lowest_first
        return self._lowest_first_strategy(legal_actions, observation)
    
    def _clear_hand_strategy(self, legal_actions: np.ndarray, observation: np.ndarray) -> int:
        """Strategy: Try to play as many cards as possible to clear hand quickly."""
        best_action = None
        most_cards = 0
        
        for action_id in legal_actions:
            action_spec = self.action_space.get_action_spec(action_id)
            
            if action_spec.hand_type == HandType.PASS:
                continue  # Skip pass unless nothing else available
            
            num_cards = len(action_spec.card_indices)
            
            if num_cards > most_cards:
                most_cards = num_cards
                best_action = action_id
            elif num_cards == most_cards and action_spec.card_indices:
                # Tie-breaker: prefer lower card indices
                current_min = min(self.action_space.get_action_spec(best_action).card_indices) if best_action else float('inf')
                this_min = min(action_spec.card_indices)
                if this_min < current_min:
                    best_action = action_id
        
        return int(best_action) if best_action is not None else int(legal_actions[0])
    
    def _adaptive_strategy(self, legal_actions: np.ndarray, observation: np.ndarray) -> int:
        """Strategy: Adapt based on game context.
        
        This attempts to use game information to make better decisions.
        """
        # Try to extract game context from observation
        # Note: This is simplified - real implementation would need proper observation parsing
        
        # For now, use a hybrid approach:
        # - Early game: play lowest cards
        # - Late game: play to clear hand quickly
        
        # Heuristic: if we have many legal actions, we're probably early game
        if len(legal_actions) > 50:
            return self._lowest_first_strategy(legal_actions, observation)
        elif len(legal_actions) > 20:
            return self._clear_hand_strategy(legal_actions, observation)
        else:
            # Few options available - be more aggressive
            return self._singles_first_strategy(legal_actions, observation)
    
    def reset(self) -> None:
        """Reset agent state."""
        pass  # Stateless agent
    
    def set_strategy(self, strategy: str) -> None:
        """Change the greedy strategy.
        
        Args:
            strategy: New strategy name
        """
        if strategy not in self.strategy_functions:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.strategy_functions.keys())}")
        self.strategy = strategy
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about current strategy.
        
        Returns:
            Dictionary with strategy information
        """
        return {
            'agent_type': 'FixedActionGreedyAgent',
            'current_strategy': self.strategy,
            'available_strategies': list(self.strategy_functions.keys()),
            'strategy_description': self._get_strategy_description(self.strategy)
        }
    
    def _get_strategy_description(self, strategy: str) -> str:
        """Get description of strategy."""
        descriptions = {
            "lowest_first": "Play lowest/weakest cards first to get rid of them",
            "highest_first": "Play highest/strongest cards first (generally poor strategy)",
            "singles_first": "Strongly prefer singles to clear hand quickly",
            "clear_hand": "Try to play as many cards as possible each turn",
            "adaptive": "Adapt strategy based on game context and available actions",
        }
        return descriptions.get(strategy, "Unknown strategy")


class FixedActionSmartGreedyAgent(FixedActionGreedyAgent):
    """Enhanced greedy agent with more sophisticated decision making.
    
    This agent tries to make smarter decisions by considering hand structure
    and game context more carefully.
    """
    
    def __init__(self, name: str = "SmartGreedy"):
        super().__init__(name, strategy="adaptive")
        self.hand_memory = {}  # Remember hand structures seen
    
    def get_action(
        self, 
        observation: np.ndarray, 
        action_mask: Optional[np.ndarray] = None
    ) -> int:
        """Enhanced action selection with hand analysis."""
        if action_mask is None:
            return 0
        
        legal_actions = np.where(action_mask)[0]
        if len(legal_actions) == 0:
            return 0
        
        # Analyze available actions by type
        action_analysis = self._analyze_legal_actions(legal_actions)
        
        # Make decision based on analysis
        return self._smart_decision(legal_actions, action_analysis, observation)
    
    def _analyze_legal_actions(self, legal_actions: np.ndarray) -> Dict[str, List[int]]:
        """Analyze legal actions by type.
        
        Args:
            legal_actions: Array of legal action IDs
            
        Returns:
            Dictionary categorizing actions by type
        """
        analysis = {
            'singles': [],
            'pairs': [],
            'triples': [],
            'five_cards': [],
            'passes': []
        }
        
        for action_id in legal_actions:
            action_spec = self.action_space.get_action_spec(action_id)
            
            if action_spec.hand_type == HandType.SINGLE:
                analysis['singles'].append(action_id)
            elif action_spec.hand_type == HandType.PAIR:
                analysis['pairs'].append(action_id)
            elif action_spec.hand_type == HandType.TRIPLE:
                analysis['triples'].append(action_id)
            elif action_spec.hand_type == HandType.FIVE_CARD:
                analysis['five_cards'].append(action_id)
            elif action_spec.hand_type == HandType.PASS:
                analysis['passes'].append(action_id)
        
        return analysis
    
    def _smart_decision(
        self, 
        legal_actions: np.ndarray, 
        action_analysis: Dict[str, List[int]], 
        observation: np.ndarray
    ) -> int:
        """Make smart decision based on action analysis.
        
        Args:
            legal_actions: Legal action IDs
            action_analysis: Categorized actions
            observation: Game observation
            
        Returns:
            Selected action ID
        """
        # Priority 1: If we have singles, prefer the lowest one
        if action_analysis['singles']:
            singles = action_analysis['singles']
            best_single = min(singles, key=lambda aid: min(
                self.action_space.get_action_spec(aid).card_indices
            ))
            return int(best_single)
        
        # Priority 2: If we have pairs, prefer lower pairs
        if action_analysis['pairs']:
            pairs = action_analysis['pairs']
            best_pair = min(pairs, key=lambda aid: min(
                self.action_space.get_action_spec(aid).card_indices
            ))
            return int(best_pair)
        
        # Priority 3: Consider triples if available
        if action_analysis['triples']:
            triples = action_analysis['triples']
            best_triple = min(triples, key=lambda aid: min(
                self.action_space.get_action_spec(aid).card_indices
            ))
            return int(best_triple)
        
        # Priority 4: Five-card hands (use carefully)
        if action_analysis['five_cards']:
            five_cards = action_analysis['five_cards']
            # Prefer five-card hands with lower cards
            best_five_card = min(five_cards, key=lambda aid: min(
                self.action_space.get_action_spec(aid).card_indices
            ))
            return int(best_five_card)
        
        # Last resort: Pass
        if action_analysis['passes']:
            return int(action_analysis['passes'][0])
        
        # Fallback
        return int(legal_actions[0])


# Convenience functions for creating greedy agent variants
def create_basic_greedy_agent(name: str = "BasicGreedy") -> FixedActionGreedyAgent:
    """Create basic greedy agent that plays lowest cards first."""
    return FixedActionGreedyAgent(name=name, strategy="lowest_first")


def create_aggressive_greedy_agent(name: str = "AggressiveGreedy") -> FixedActionGreedyAgent:
    """Create greedy agent that tries to clear hand quickly."""
    return FixedActionGreedyAgent(name=name, strategy="clear_hand")


def create_smart_greedy_agent(name: str = "SmartGreedy") -> FixedActionSmartGreedyAgent:
    """Create enhanced greedy agent with sophisticated decision making."""
    return FixedActionSmartGreedyAgent(name=name)