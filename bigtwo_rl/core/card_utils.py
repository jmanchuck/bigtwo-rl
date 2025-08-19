"""Card encoding/decoding utilities for human-readable format."""
import numpy as np

def card_to_string(card_idx):
    """Convert card index (0-51) to human readable format (e.g., 'AS', 'KH')."""
    rank = card_idx // 4
    suit = card_idx % 4
    
    # Ranks: 0=3, 1=4, ..., 8=T, 9=J, 10=Q, 11=K, 12=A, 13=2
    rank_chars = "3456789TJQKA2"
    suit_chars = "DCHS"  # Diamonds, Clubs, Hearts, Spades
    
    return rank_chars[rank] + suit_chars[suit]

def string_to_card(card_str):
    """Convert human readable format to card index."""
    if len(card_str) != 2:
        raise ValueError(f"Invalid card format: {card_str}")
    
    rank_char, suit_char = card_str.upper()
    
    # Map characters to indices
    rank_chars = "3456789TJQKA2"
    suit_chars = "DCHS"
    
    try:
        rank = rank_chars.index(rank_char)
        suit = suit_chars.index(suit_char)
        return rank * 4 + suit
    except ValueError:
        raise ValueError(f"Invalid card: {card_str}")

def hand_to_strings(hand):
    """Convert list of card indices to readable strings."""
    return [card_to_string(card) for card in sorted(hand)]

def strings_to_cards(card_strings):
    """Convert list of card strings to indices."""
    return [string_to_card(card_str) for card_str in card_strings]

def format_hand(hand):
    """Format hand for display."""
    cards = hand_to_strings(hand)
    return " ".join(cards)

def parse_move_input(input_str, hand):
    """Parse user input for card play."""
    if not input_str.strip() or input_str.lower() == 'pass':
        return []
    
    # Split by spaces and convert
    card_strings = input_str.strip().upper().split()
    try:
        cards = strings_to_cards(card_strings)
        # Validate cards are in hand
        for card in cards:
            if card not in hand:
                raise ValueError(f"Card {card_to_string(card)} not in hand")
        return sorted(cards)
    except ValueError as e:
        raise ValueError(f"Invalid input: {e}")

# Numpy array conversion functions

def hand_array_to_strings(hand_array):
    """Convert numpy boolean array (shape 52) to readable strings."""
    card_indices = np.where(hand_array)[0]
    return [card_to_string(card) for card in sorted(card_indices)]

def strings_to_hand_array(card_strings):
    """Convert list of card strings to numpy boolean array."""
    hand_array = np.zeros(52, dtype=bool)
    card_indices = strings_to_cards(card_strings)
    hand_array[card_indices] = True
    return hand_array

def format_hand_array(hand_array):
    """Format numpy hand array for display."""
    cards = hand_array_to_strings(hand_array)
    return " ".join(cards)

def parse_move_input_array(input_str, hand_array):
    """Parse user input for card play with numpy hand array."""
    if not input_str.strip() or input_str.lower() == 'pass':
        return np.zeros(52, dtype=bool)  # Return empty move array
    
    # Split by spaces and convert
    card_strings = input_str.strip().upper().split()
    try:
        cards = strings_to_cards(card_strings)
        # Validate cards are in hand
        for card in cards:
            if not hand_array[card]:
                raise ValueError(f"Card {card_to_string(card)} not in hand")
        
        # Create move array
        move_array = np.zeros(52, dtype=bool)
        move_array[cards] = True
        return move_array
    except ValueError as e:
        raise ValueError(f"Invalid input: {e}")

def card_indices_from_array(hand_array):
    """Extract card indices from numpy boolean array."""
    return np.where(hand_array)[0]

def array_from_card_indices(card_indices):
    """Create numpy boolean array from card indices."""
    hand_array = np.zeros(52, dtype=bool)
    hand_array[card_indices] = True
    return hand_array