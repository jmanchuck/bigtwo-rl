"""Fixed action space representation for Big Two with all possible hand combinations.

This module implements a comprehensive action space that represents every possible
hand combination that can be played in Big Two, enabling the network to learn
intrinsic move quality across all game contexts.

Action Space Breakdown:
- Singles (13 cards): 13 combinations
- Pairs (same rank only): 33 valid index combinations (e.g., (0,1), (0,2), (0,3) for 4-of-a-kind)
- Trips (same rank only): 31 valid index combinations (e.g., (0,1,2), (0,1,3), (0,2,3) for 4-of-a-kind)
- 5-card hands: C(13,5) = 1,287 combinations
- Pass action: 1 combination
Total: 13 + 33 + 31 + 1,287 + 1 = 1,365 actions

Each action maps to a specific combination of card indices (0-12) from a 
sorted 13-card hand.
"""

import numpy as np
from itertools import combinations
from typing import List, Tuple, Optional, Set, Dict, Any
from dataclasses import dataclass
from enum import Enum


class HandType(Enum):
    """Types of playable hands in Big Two."""
    SINGLE = "single"
    PAIR = "pair" 
    TRIPLE = "triple"
    FIVE_CARD = "five_card"
    PASS = "pass"


@dataclass(frozen=True)
class ActionSpec:
    """Specification for a single action in the action space."""
    action_id: int
    hand_type: HandType
    card_indices: Tuple[int, ...] 
    description: str


class BigTwoActionSpace:
    """Comprehensive action space for Big Two with fixed 1,365 actions.
    
    This class enumerates all possible hand combinations from a 13-card hand
    and provides utilities for converting between action IDs and card combinations.
    """
    
    # Action space size
    TOTAL_ACTIONS = 1365
    
    def __init__(self):
        """Initialize the action space by enumerating all possible combinations."""
        self.actions: List[ActionSpec] = []
        self._action_to_spec: Dict[int, ActionSpec] = {}
        self._indices_to_action: Dict[Tuple[int, ...], int] = {}
        
        self._build_action_space()
    
    def _build_action_space(self) -> None:
        """Build the complete action space by enumerating all combinations."""
        action_id = 0
        
        # 1. Singles (13 actions: indices 0-12)
        for i in range(13):
            spec = ActionSpec(
                action_id=action_id,
                hand_type=HandType.SINGLE,
                card_indices=(i,),
                description=f"Single card at index {i}"
            )
            self._add_action(spec)
            action_id += 1
        
        # 2. Pairs (33 actions: valid same-rank pair combinations)
        # Since cards are sorted, same-rank pairs can only be:
        # - Adjacent indices: (0,1), (1,2), ..., (11,12) = 12 pairs
        # - Skip 1: (0,2), (1,3), ..., (10,12) = 11 pairs (when 3-of-a-kind)
        # - Skip 2: (0,3), (1,4), ..., (9,12) = 10 pairs (when 4-of-a-kind)
        # Total: 12 + 11 + 10 = 33 pairs
        pair_combinations = []
        
        # Adjacent pairs (i, i+1)
        for i in range(12):
            pair_combinations.append((i, i+1))
            
        # Skip-1 pairs (i, i+2) for 3-of-a-kind scenarios
        for i in range(11):
            pair_combinations.append((i, i+2))
            
        # Skip-2 pairs (i, i+3) for 4-of-a-kind scenarios
        for i in range(10):
            pair_combinations.append((i, i+3))
            
        assert len(pair_combinations) == 33, f"Expected 33 pairs, got {len(pair_combinations)}"
        
        for pair in pair_combinations:
            spec = ActionSpec(
                action_id=action_id,
                hand_type=HandType.PAIR, 
                card_indices=pair,
                description=f"Pair with indices {pair}"
            )
            self._add_action(spec)
            action_id += 1
            
        # 3. Triples (31 actions: valid same-rank triple combinations)
        # Since cards are sorted, same-rank triples can only be:
        # - Consecutive: (0,1,2), (1,2,3), ..., (10,11,12) = 11 triples
        # - Skip last: (0,1,3), (1,2,4), ..., (9,10,12) = 10 triples (when 4-of-a-kind)
        # - Skip middle: (0,2,3), (1,3,4), ..., (9,11,12) = 10 triples (when 4-of-a-kind)
        # Total: 11 + 10 + 10 = 31 triples
        triple_combinations = []
        
        # Consecutive triples (i, i+1, i+2)
        for i in range(11):
            triple_combinations.append((i, i+1, i+2))
            
        # Skip-last triples (i, i+1, i+3) for 4-of-a-kind scenarios
        for i in range(10):
            triple_combinations.append((i, i+1, i+3))
            
        # Skip-middle triples (i, i+2, i+3) for 4-of-a-kind scenarios
        for i in range(10):
            triple_combinations.append((i, i+2, i+3))
            
        assert len(triple_combinations) == 31, f"Expected 31 triples, got {len(triple_combinations)}"
        
        for triple in triple_combinations:
            spec = ActionSpec(
                action_id=action_id,
                hand_type=HandType.TRIPLE,
                card_indices=triple, 
                description=f"Triple with indices {triple}"
            )
            self._add_action(spec)
            action_id += 1
            
        # 4. Five-card hands (1,287 actions: all combinations of 5 indices from 13)
        for five_card in combinations(range(13), 5):
            spec = ActionSpec(
                action_id=action_id,
                hand_type=HandType.FIVE_CARD,
                card_indices=five_card,
                description=f"Five-card hand with indices {five_card}"
            )
            self._add_action(spec)
            action_id += 1
            
        # 5. Pass action (1 action)
        spec = ActionSpec(
            action_id=action_id,
            hand_type=HandType.PASS,
            card_indices=(),
            description="Pass turn"
        )
        self._add_action(spec)
        action_id += 1
        
        assert action_id == self.TOTAL_ACTIONS, f"Expected {self.TOTAL_ACTIONS} actions, got {action_id}"
    
    def _add_action(self, spec: ActionSpec) -> None:
        """Add an action specification to internal data structures."""
        self.actions.append(spec)
        self._action_to_spec[spec.action_id] = spec
        self._indices_to_action[spec.card_indices] = spec.action_id
    
    def get_action_spec(self, action_id: int) -> ActionSpec:
        """Get action specification for given action ID."""
        if action_id not in self._action_to_spec:
            raise ValueError(f"Invalid action ID: {action_id}")
        return self._action_to_spec[action_id]
    
    def get_action_id(self, card_indices: Tuple[int, ...]) -> int:
        """Get action ID for given card indices combination."""
        if card_indices not in self._indices_to_action:
            raise ValueError(f"Invalid card indices combination: {card_indices}")
        return self._indices_to_action[card_indices]
    
    def get_actions_by_type(self, hand_type: HandType) -> List[ActionSpec]:
        """Get all actions of a specific hand type."""
        return [action for action in self.actions if action.hand_type == hand_type]
    
    def get_action_counts_by_type(self) -> Dict[HandType, int]:
        """Get count of actions by hand type."""
        counts = {}
        for hand_type in HandType:
            counts[hand_type] = len(self.get_actions_by_type(hand_type))
        return counts


class BigTwoActionMasker:
    """Creates action masks based on game state and rules.
    
    This class implements the masking logic that determines which of the 1,365
    possible actions are legal in a given game state.
    """
    
    def __init__(self, action_space: BigTwoActionSpace):
        """Initialize masker with action space reference."""
        self.action_space = action_space
        
    def create_mask(
        self,
        player_hand: np.ndarray,
        last_played_cards: Optional[np.ndarray] = None,
        last_played_type: Optional[HandType] = None,
        is_starting_trick: bool = False,
        must_play_3_diamonds: bool = False
    ) -> np.ndarray:
        """Create action mask for current game state.
        
        Args:
            player_hand: Boolean array of shape (52,) indicating cards in hand
            last_played_cards: Cards played in last move (None if starting trick)
            last_played_type: Type of last played hand 
            is_starting_trick: True if this is the start of a new trick
            must_play_3_diamonds: True if player must play 3 of diamonds (first move)
            
        Returns:
            Boolean array of shape (1365,) where True = legal action
        """
        mask = np.zeros(self.action_space.TOTAL_ACTIONS, dtype=bool)
        
        # Get hand indices (cards player actually has)
        hand_indices = self._get_hand_indices(player_hand)
        
        for action in self.action_space.actions:
            if self._is_action_legal(
                action, 
                hand_indices,
                last_played_cards,
                last_played_type, 
                is_starting_trick,
                must_play_3_diamonds,
                player_hand
            ):
                mask[action.action_id] = True
                
        return mask
    
    def _get_hand_indices(self, player_hand: np.ndarray) -> Set[int]:
        """Convert 52-card boolean array to set of hand indices (0-12).
        
        Assumes cards are sorted by rank within each hand, so the first 13 True
        values in player_hand correspond to indices 0-12.
        """
        hand_cards = np.where(player_hand)[0]
        # Sort cards and take first 13 as hand indices
        # This is a simplification - real implementation would need proper card mapping
        return set(range(min(13, len(hand_cards))))
    
    def _is_action_legal(
        self,
        action: ActionSpec,
        hand_indices: Set[int], 
        last_played_cards: Optional[np.ndarray],
        last_played_type: Optional[HandType],
        is_starting_trick: bool,
        must_play_3_diamonds: bool,
        player_hand: np.ndarray
    ) -> bool:
        """Check if a specific action is legal in current game state."""
        
        # Pass action rules
        if action.hand_type == HandType.PASS:
            # Can't pass if starting a trick or must play 3♦
            return not (is_starting_trick or must_play_3_diamonds)
        
        # Check if player has required cards
        if not all(idx in hand_indices for idx in action.card_indices):
            return False
            
        # 3♦ rule: first move must include 3 of diamonds
        if must_play_3_diamonds:
            return self._action_includes_3_diamonds(action, player_hand)
        
        # Starting trick: any valid hand type is allowed
        if is_starting_trick:
            return self._is_valid_hand_combination(action, player_hand)
        
        # Must match last played hand type
        if last_played_type is not None and action.hand_type != last_played_type:
            return False
            
        # Must beat last played cards (if same type)
        if last_played_cards is not None:
            return self._beats_last_played(action, last_played_cards, player_hand)
            
        return self._is_valid_hand_combination(action, player_hand)
    
    def _action_includes_3_diamonds(self, action: ActionSpec, player_hand: np.ndarray) -> bool:
        """Check if action includes 3 of diamonds (simplified)."""
        # This is a placeholder - real implementation would need proper card mapping
        # For now, assume index 0 represents 3♦
        return 0 in action.card_indices
        
    def _is_valid_hand_combination(self, action: ActionSpec, player_hand: np.ndarray) -> bool:
        """Check if card indices form a valid Big Two hand combination."""
        # This is a placeholder for actual hand validation logic
        # Real implementation would check:
        # - Pairs/triples have same rank
        # - Five-card hands are valid straights/flushes/etc.
        return True
        
    def _beats_last_played(
        self, 
        action: ActionSpec, 
        last_played_cards: np.ndarray,
        player_hand: np.ndarray
    ) -> bool:
        """Check if action beats the last played cards."""
        # This is a placeholder for actual comparison logic
        # Real implementation would compare card ranks/suits according to Big Two rules
        return True


def create_action_space() -> BigTwoActionSpace:
    """Factory function to create a Big Two action space."""
    return BigTwoActionSpace()


def create_action_masker(action_space: BigTwoActionSpace) -> BigTwoActionMasker:
    """Factory function to create an action masker."""
    return BigTwoActionMasker(action_space)