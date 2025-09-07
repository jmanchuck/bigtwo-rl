"""Core card representation and encoding for Big Two."""

# ============================================================
# Big Two Card System
# ============================================================

# Ranks: 3(0) < 4(1) < ... < A(11) < 2(12)
RANKS = list(range(13))  # 0..12
SUITS = list(range(4))  # 0..3 (♦,♣,♥,♠)

# All 52 cards encoded
ALL_CARDS = [r << 2 | s for r in RANKS for s in SUITS]

# Special cards
THREE_DIAMONDS = 0 << 2 | 0  # rank=0, suit=0


def encode(rank: int, suit: int) -> int:
    """Encode card as one byte: (rank<<2)|suit."""
    return rank << 2 | suit


def rank_of(card_code: int) -> int:
    """Extract rank from encoded card."""
    return card_code >> 2


def suit_of(card_code: int) -> int:
    """Extract suit from encoded card."""
    return card_code & 3


def card_to_string(card_code: int) -> str:
    """Convert encoded card to string like '3D', 'AS', etc."""
    rank = rank_of(card_code)
    suit = suit_of(card_code)

    rank_chars = "3456789TJQKA2"
    suit_chars = "DCHS"  # ♦♣♥♠

    return rank_chars[rank] + suit_chars[suit]


def string_to_card(card_str: str) -> int:
    """Parse card string like '3D', 'AS' into encoded card."""
    rank_char = card_str[0]
    suit_char = card_str[1]

    rank_map = {
        "3": 0,
        "4": 1,
        "5": 2,
        "6": 3,
        "7": 4,
        "8": 5,
        "9": 6,
        "T": 7,
        "J": 8,
        "Q": 9,
        "K": 10,
        "A": 11,
        "2": 12,
    }
    suit_map = {"D": 0, "C": 1, "H": 2, "S": 3}

    return encode(rank_map[rank_char], suit_map[suit_char])
