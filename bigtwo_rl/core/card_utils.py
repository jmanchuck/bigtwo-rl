"""Card encoding/decoding utilities for human-readable format."""

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