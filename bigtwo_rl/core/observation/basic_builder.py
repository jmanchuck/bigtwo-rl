"""Basic observation builder implementation."""

from typing import List, Dict, Any
import numpy as np

from ..cards import rank_of, suit_of
from ..game.types import Hand
from .observation_builder import ObservationBuilder


class BasicObservationBuilder(ObservationBuilder):
    """
    Basic implementation of ObservationBuilder interface.

    Uses 168-bit observation vector:
    - 52 bits: current player's hand
    - 52 bits: last played cards
    - 64 bits: game state (card counts + flags)
    """

    def __init__(self):
        super().__init__(observation_size=168)  # 52 + 52 + 64

    def encode_hand(self, hand: Hand) -> np.ndarray:
        """Encode hand as 52-bit vector where each bit represents card presence."""
        hand_bits = np.zeros(52, dtype=np.bool_)

        for i in range(13):
            if hand.played[i]:
                continue  # Skip played cards

            card_code = hand.card[i]
            rank = rank_of(card_code)
            suit = suit_of(card_code)

            # Map to standard deck index: rank * 4 + suit
            deck_idx = rank * 4 + suit
            hand_bits[deck_idx] = True

        return hand_bits

    def encode_last_play(self, last_played_cards: List[int], passes: int) -> np.ndarray:
        """Encode last play as 52-bit vector. All zeros if no last play or 3 passes."""
        play_bits = np.zeros(52, dtype=np.bool_)

        # If no last play (start of game) or 3 passes in a row, return all zeros
        if not last_played_cards or passes >= 3:
            return play_bits

        # Set bits for each played card
        for card_code in last_played_cards:
            rank = rank_of(card_code)
            suit = suit_of(card_code)
            deck_idx = rank * 4 + suit
            play_bits[deck_idx] = True

        return play_bits

    def encode_game_state(
        self, current_player: int, player_card_counts: List[int], passes: int, is_first_play: bool
    ) -> np.ndarray:
        """Encode game state as one-hot encoding of card counts + game flags (4×14 + 8 bits = 64 bits)."""
        # Card counts: 4 players × 14 possible counts (0-13 cards) = 56 bits
        card_count_bits = np.zeros(56, dtype=np.bool_)
        for player_idx in range(4):
            card_count = min(player_card_counts[player_idx], 13)  # Cap at 13 for one-hot encoding
            bit_position = player_idx * 14 + card_count
            card_count_bits[bit_position] = True

        # Game flags: 8 bits for current player (one-hot) + passes + is_first_play
        flag_bits = np.zeros(8, dtype=np.bool_)
        flag_bits[current_player] = True  # One-hot encode current player (4 bits)

        # Encode passes (0-3) in 2 bits
        if passes >= 1:
            flag_bits[4] = True
        if passes >= 2:
            flag_bits[5] = True
        if passes >= 3:
            flag_bits[6] = True

        flag_bits[7] = is_first_play  # First play flag

        return np.concatenate([card_count_bits, flag_bits])

    def get_feature_info(self) -> Dict[str, Any]:
        return {
            "hand_features": {"size": 52, "description": "Binary presence of each card in current player's hand"},
            "last_play_features": {
                "size": 52,
                "description": "Binary representation of last played cards (all zeros if no play or 3 passes)",
            },
            "game_state_features": {
                "size": 64,
                "description": "Card counts (4 players × 14 counts = 56 bits) + game flags (8 bits: current player, passes, first play)",
            },
            "total_size": 168,
            "deck_mapping": "Cards mapped as: rank * 4 + suit (rank 0-12, suit 0-3)",
        }
