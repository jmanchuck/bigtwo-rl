"""Card mapping system for translating between game representation and action space.

This module handles the complex mapping between the game's 52-card representation
and the action space's sorted 0-12 indices representation.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from .bigtwo import ToyBigTwoFullRules
from .action_space import BigTwoActionSpace, HandType


class CardMapper:
    """Handles mapping between game cards and sorted hand indices."""
    
    def __init__(self):
        self.game = ToyBigTwoFullRules()  # For card utilities
        
    def game_hand_to_sorted_indices(self, hand_52: np.ndarray) -> Dict[int, int]:
        """Map 52-card hand to sorted 0-12 indices.
        
        Args:
            hand_52: Boolean array of shape (52,) indicating cards in hand
            
        Returns:
            Dict mapping sorted_index -> game_card_id
        """
        # Get player's cards
        player_cards = np.where(hand_52)[0].tolist()
        
        if not player_cards:
            return {}
        
        # Sort by Big Two rules (value, then suit)
        def card_key(card_id):
            return (self.game._VALUE_TABLE[card_id], self.game._SUIT_TABLE[card_id])
            
        sorted_cards = sorted(player_cards, key=card_key)
        
        # Create mapping: sorted index -> actual card ID
        return {i: card_id for i, card_id in enumerate(sorted_cards)}
        
    def indices_to_game_cards(self, indices: Tuple[int, ...], hand_52: np.ndarray) -> np.ndarray:
        """Convert sorted indices back to 52-card move array.
        
        Args:
            indices: Tuple of sorted indices from action space
            hand_52: Player's hand as 52-card boolean array
            
        Returns:
            Boolean array of shape (52,) representing the move
        """
        if not indices:  # Pass move
            return np.zeros(52, dtype=bool)
            
        # Get the mapping
        index_to_card = self.game_hand_to_sorted_indices(hand_52)
        
        # Create move mask
        move_mask = np.zeros(52, dtype=bool)
        for idx in indices:
            if idx in index_to_card:
                card_id = index_to_card[idx]
                move_mask[card_id] = True
                
        return move_mask
        
    def validate_action_feasible(self, action_id: int, hand_52: np.ndarray, action_space: BigTwoActionSpace) -> bool:
        """Check if action is feasible given current hand.
        
        Args:
            action_id: Action ID from 0-1364
            hand_52: Player's hand as 52-card boolean array
            action_space: Action space instance
            
        Returns:
            True if action can be executed with current hand
        """
        action_spec = action_space.get_action_spec(action_id)
        
        if action_spec.hand_type == HandType.PASS:
            return True
            
        # Check if player has enough cards for this action
        player_cards = np.where(hand_52)[0]
        if len(player_cards) < len(action_spec.card_indices):
            return False
            
        # Check if the specific indices exist in sorted hand
        index_to_card = self.game_hand_to_sorted_indices(hand_52)
        return all(idx in index_to_card for idx in action_spec.card_indices)
    
    def get_valid_actions_for_hand(self, hand_52: np.ndarray, action_space: BigTwoActionSpace) -> List[int]:
        """Get list of action IDs that are feasible for the given hand.
        
        Args:
            hand_52: Player's hand as 52-card boolean array
            action_space: Action space instance
            
        Returns:
            List of valid action IDs
        """
        valid_actions = []
        
        for action in action_space.actions:
            if self.validate_action_feasible(action.action_id, hand_52, action_space):
                valid_actions.append(action.action_id)
                
        return valid_actions
    
    def get_hand_signature(self, hand_52: np.ndarray) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Get hand signature for caching and comparison.
        
        Args:
            hand_52: Player's hand as 52-card boolean array
            
        Returns:
            Tuple of (ranks, suits) representing the hand structure
        """
        player_cards = np.where(hand_52)[0].tolist()
        
        if not player_cards:
            return ((), ())
        
        # Sort cards and extract ranks and suits
        def card_key(card_id):
            return (self.game._VALUE_TABLE[card_id], self.game._SUIT_TABLE[card_id])
            
        sorted_cards = sorted(player_cards, key=card_key)
        
        ranks = tuple(self.game._RANK_TABLE[card] for card in sorted_cards)
        suits = tuple(self.game._SUIT_TABLE[card] for card in sorted_cards)
        
        return (ranks, suits)
    
    def debug_hand_mapping(self, hand_52: np.ndarray) -> Dict:
        """Debug information about hand mapping.
        
        Args:
            hand_52: Player's hand as 52-card boolean array
            
        Returns:
            Dictionary with debugging information
        """
        player_cards = np.where(hand_52)[0].tolist()
        index_to_card = self.game_hand_to_sorted_indices(hand_52)
        
        debug_info = {
            'num_cards': len(player_cards),
            'raw_cards': player_cards,
            'sorted_mapping': index_to_card,
            'hand_signature': self.get_hand_signature(hand_52)
        }
        
        # Add card details
        card_details = []
        for idx, card_id in index_to_card.items():
            rank = self.game._RANK_TABLE[card_id]
            suit = self.game._SUIT_TABLE[card_id]
            value = self.game._VALUE_TABLE[card_id]
            
            # Convert to human readable
            rank_names = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
            suit_names = ['♦', '♣', '♥', '♠']
            
            card_details.append({
                'index': idx,
                'card_id': card_id,
                'rank': rank,
                'suit': suit,
                'value': value,
                'display': f"{rank_names[rank]}{suit_names[suit]}"
            })
            
        debug_info['card_details'] = card_details
        
        return debug_info


class ActionTranslator:
    """Translates between different action representations."""
    
    def __init__(self):
        self.card_mapper = CardMapper()
        self.action_space = BigTwoActionSpace()
    
    def game_move_to_action_id(self, move_mask: np.ndarray, hand_52: np.ndarray) -> int:
        """Convert game move to action ID.
        
        Args:
            move_mask: 52-card boolean array representing the move
            hand_52: Player's hand as 52-card boolean array
            
        Returns:
            Action ID corresponding to the move
        """
        if not np.any(move_mask):
            # Pass move
            return self.action_space.get_action_id(())
        
        # Get cards in the move
        move_cards = np.where(move_mask)[0].tolist()
        
        # Map to sorted indices
        index_to_card = self.card_mapper.game_hand_to_sorted_indices(hand_52)
        card_to_index = {card_id: idx for idx, card_id in index_to_card.items()}
        
        # Convert move cards to sorted indices
        try:
            sorted_indices = tuple(sorted([card_to_index[card_id] for card_id in move_cards]))
            return self.action_space.get_action_id(sorted_indices)
        except KeyError:
            raise ValueError(f"Move contains cards not in hand: {move_cards}")
    
    def action_id_to_game_move(self, action_id: int, hand_52: np.ndarray) -> np.ndarray:
        """Convert action ID to game move.
        
        Args:
            action_id: Action ID from 0-1364
            hand_52: Player's hand as 52-card boolean array
            
        Returns:
            52-card boolean array representing the move
        """
        action_spec = self.action_space.get_action_spec(action_id)
        return self.card_mapper.indices_to_game_cards(action_spec.card_indices, hand_52)
    
    def legacy_action_to_fixed_action(self, legacy_action_index: int, legal_moves: List[np.ndarray], hand_52: np.ndarray) -> int:
        """Convert legacy action index to fixed action ID.
        
        Args:
            legacy_action_index: Index into legal_moves list
            legal_moves: List of legal moves from game engine
            hand_52: Player's hand as 52-card boolean array
            
        Returns:
            Fixed action ID
        """
        if legacy_action_index >= len(legal_moves):
            # Was a pass in old system
            return self.action_space.get_action_id(())
        
        # Get the actual move
        selected_move = legal_moves[legacy_action_index]
        
        # Convert to action ID
        return self.game_move_to_action_id(selected_move, hand_52)
    
    def fixed_action_to_legacy_action(self, action_id: int, legal_moves: List[np.ndarray], hand_52: np.ndarray) -> int:
        """Convert fixed action ID to legacy action index.
        
        Args:
            action_id: Fixed action ID
            legal_moves: List of legal moves from game engine
            hand_52: Player's hand as 52-card boolean array
            
        Returns:
            Legacy action index
        """
        action_spec = self.action_space.get_action_spec(action_id)
        
        if action_spec.hand_type == HandType.PASS:
            return len(legal_moves)  # Pass was always last index in old system
        
        # Convert to game move
        game_move = self.card_mapper.indices_to_game_cards(action_spec.card_indices, hand_52)
        
        # Find matching move in legal_moves
        for i, legal_move in enumerate(legal_moves):
            if np.array_equal(game_move, legal_move):
                return i
        
        # No exact match found - this shouldn't happen if action is legal
        raise ValueError(f"Could not find matching legal move for action {action_id}")