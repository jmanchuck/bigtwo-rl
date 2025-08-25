"""Enhanced Big Two action system for fixed 1,365-action space.

This module implements a comprehensive action system that bridges the fixed action space
with the existing game engine, providing translation, masking, and validation capabilities.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from .action_space import BigTwoActionSpace, BigTwoActionMasker, HandType, ActionSpec
from .bigtwo import ToyBigTwoFullRules


class CardMapper:
    """Handles mapping between game cards and sorted hand indices."""
    
    def __init__(self):
        self.game = ToyBigTwoFullRules()  # For card utilities
        
    def game_hand_to_sorted_indices(self, hand_52: np.ndarray) -> Dict[int, int]:
        """Map 52-card hand to sorted 0-12 indices.
        
        Returns:
            Dict mapping sorted_index -> game_card_id
        """
        # Get player's cards
        player_cards = np.where(hand_52)[0].tolist()
        
        # Sort by Big Two rules (value, then suit)
        def card_key(card_id):
            return (self.game._VALUE_TABLE[card_id], self.game._SUIT_TABLE[card_id])
            
        sorted_cards = sorted(player_cards, key=card_key)
        
        # Create mapping: sorted index -> actual card ID
        return {i: card_id for i, card_id in enumerate(sorted_cards)}
        
    def indices_to_game_cards(self, indices: Tuple[int, ...], hand_52: np.ndarray) -> np.ndarray:
        """Convert sorted indices back to 52-card move array."""
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
        """Check if action is feasible given current hand."""
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


class EnhancedActionMasker(BigTwoActionMasker):
    """Enhanced action masker with proper game rules integration."""
    
    def __init__(self, action_space: BigTwoActionSpace):
        super().__init__(action_space)
        self.card_mapper = CardMapper()
        self.game = ToyBigTwoFullRules()
    
    def create_mask(
        self,
        player_hand: np.ndarray,
        last_played_cards: Optional[np.ndarray] = None,
        last_played_type: Optional[HandType] = None,
        is_starting_trick: bool = False,
        must_play_3_diamonds: bool = False
    ) -> np.ndarray:
        """Create action mask for current game state with proper game rules."""
        mask = np.zeros(self.action_space.TOTAL_ACTIONS, dtype=bool)
        
        # Get sorted hand mapping
        index_to_card = self.card_mapper.game_hand_to_sorted_indices(player_hand)
        num_cards = len(index_to_card)
        
        if num_cards == 0:
            return mask  # No cards, no valid actions
        
        for action in self.action_space.actions:
            if self._is_action_legal_enhanced(
                action, 
                index_to_card,
                player_hand,
                last_played_cards,
                last_played_type, 
                is_starting_trick,
                must_play_3_diamonds
            ):
                mask[action.action_id] = True
                
        return mask
    
    def _is_action_legal_enhanced(
        self,
        action: ActionSpec,
        index_to_card: Dict[int, int],
        player_hand: np.ndarray,
        last_played_cards: Optional[np.ndarray],
        last_played_type: Optional[HandType],
        is_starting_trick: bool,
        must_play_3_diamonds: bool
    ) -> bool:
        """Enhanced legality check with proper game rules."""
        
        # Pass action rules
        if action.hand_type == HandType.PASS:
            # Can't pass if starting a trick or must play 3♦
            return not (is_starting_trick or must_play_3_diamonds)
        
        # Check if player has required cards
        if not all(idx in index_to_card for idx in action.card_indices):
            return False
            
        # Convert action to actual game move
        game_move = self.card_mapper.indices_to_game_cards(action.card_indices, player_hand)
        if not np.any(game_move):
            return False
            
        # 3♦ rule: first move must include 3 of diamonds (card ID 0)
        if must_play_3_diamonds:
            return game_move[0]  # 3 of diamonds is card 0
        
        # Starting trick: check if it's a valid hand combination
        if is_starting_trick:
            return self._is_valid_hand_type(game_move, action.hand_type)
        
        # Must match last played hand type and beat it
        if last_played_type is not None:
            if action.hand_type != last_played_type:
                return False
                
            if last_played_cards is not None:
                return self._beats_last_played_enhanced(game_move, last_played_cards)
                
        return self._is_valid_hand_type(game_move, action.hand_type)
    
    def _is_valid_hand_type(self, game_move: np.ndarray, expected_type: HandType) -> bool:
        """Check if game move matches expected hand type."""
        cards = np.where(game_move)[0].tolist()
        num_cards = len(cards)
        
        if expected_type == HandType.SINGLE:
            return num_cards == 1
        elif expected_type == HandType.PAIR:
            return num_cards == 2 and self._is_valid_pair(cards)
        elif expected_type == HandType.TRIPLE:
            return num_cards == 3 and self._is_valid_triple(cards)
        elif expected_type == HandType.FIVE_CARD:
            return num_cards == 5 and self._is_valid_five_card_hand(cards)
        
        return False
    
    def _is_valid_pair(self, cards: List[int]) -> bool:
        """Check if cards form a valid pair (same rank)."""
        if len(cards) != 2:
            return False
        rank1 = self.game._RANK_TABLE[cards[0]]
        rank2 = self.game._RANK_TABLE[cards[1]]
        return rank1 == rank2
    
    def _is_valid_triple(self, cards: List[int]) -> bool:
        """Check if cards form a valid triple (same rank)."""
        if len(cards) != 3:
            return False
        ranks = [self.game._RANK_TABLE[card] for card in cards]
        return all(rank == ranks[0] for rank in ranks)
    
    def _is_valid_five_card_hand(self, cards: List[int]) -> bool:
        """Check if cards form a valid 5-card hand (straight, flush, etc.)."""
        if len(cards) != 5:
            return False
        
        # Use game engine's existing hand type identification
        move_mask = np.zeros(52, dtype=bool)
        move_mask[cards] = True
        
        # Check if it's a valid 5-card hand using existing logic
        try:
            hand_type = self.game._identify_hand_type_vectorized(move_mask)
            valid_types = ['straight', 'flush', 'full_house', 'four_of_a_kind', 'straight_flush']
            return hand_type in valid_types
        except:
            return False
    
    def _beats_last_played_enhanced(self, game_move: np.ndarray, last_played_cards: np.ndarray) -> bool:
        """Check if game move beats last played cards."""
        try:
            return self.game._beats_move_vectorized(game_move, last_played_cards)
        except:
            return False


class BigTwoActionSystem:
    """Central system for managing fixed 1,365-action space."""
    
    def __init__(self):
        self.action_space = BigTwoActionSpace()
        self.masker = EnhancedActionMasker(self.action_space)
        self.card_mapper = CardMapper()
        
    def translate_action_to_game_move(self, action_id: int, player_hand: np.ndarray) -> np.ndarray:
        """Convert action ID to actual game move (52-card boolean array)."""
        action_spec = self.action_space.get_action_spec(action_id)
        
        if action_spec.hand_type == HandType.PASS:
            return np.zeros(52, dtype=bool)  # Pass move
            
        # Map sorted hand indices to actual card positions
        return self.card_mapper.indices_to_game_cards(
            action_spec.card_indices, 
            player_hand
        )
        
    def get_legal_action_mask(self, game_state, player_hand: np.ndarray) -> np.ndarray:
        """Get 1,365-dimensional mask for current game state."""
        return self.masker.create_mask(
            player_hand=player_hand,
            last_played_cards=game_state.last_play[0] if game_state.last_play else None,
            last_played_type=self._infer_hand_type(game_state.last_play),
            is_starting_trick=(game_state.passes_in_row == 3),
            must_play_3_diamonds=self._must_play_3_diamonds(game_state)
        )
    
    def _infer_hand_type(self, last_play) -> Optional[HandType]:
        """Infer hand type from last play."""
        if last_play is None:
            return None
            
        cards = np.where(last_play[0])[0]
        num_cards = len(cards)
        
        if num_cards == 1:
            return HandType.SINGLE
        elif num_cards == 2:
            return HandType.PAIR
        elif num_cards == 3:
            return HandType.TRIPLE
        elif num_cards == 5:
            return HandType.FIVE_CARD
        else:
            return None
    
    def _must_play_3_diamonds(self, game_state) -> bool:
        """Check if player must play 3 of diamonds."""
        # First move of the game
        return (game_state.last_play is None and 
                game_state.passes_in_row == 0 and
                hasattr(game_state, 'hands') and
                np.sum([np.sum(hand) for hand in game_state.hands]) == 52)
        
    def sample_masked_action(self, logits: np.ndarray, mask: np.ndarray) -> int:
        """Sample action from masked logits using proper numerical stability."""
        masked_logits = logits + (mask.astype(float) - 1.0) * 1e9
        probabilities = self._softmax(masked_logits)
        return np.random.choice(len(probabilities), p=probabilities)
        
    def get_best_masked_action(self, logits: np.ndarray, mask: np.ndarray) -> int:
        """Get deterministic best action from masked logits."""
        masked_logits = np.where(mask, logits, -np.inf)
        return np.argmax(masked_logits)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


def create_action_system() -> BigTwoActionSystem:
    """Factory function to create a Big Two action system."""
    return BigTwoActionSystem()