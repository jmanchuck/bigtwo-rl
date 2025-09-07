from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

# ============================================================
# Big Two RL — Constants and Basic Types
# ============================================================

# Ranks: 3(0) < 4(1) < ... < A(11) < 2(12)
RANKS = list(range(13))  # 0..12
SUITS = list(range(4))  # 0..3 (♦,♣,♥,♠ — pick any fixed order)


# Encode card as one byte: (rank<<2)|suit
def encode(r, s):
    return r << 2 | s


class Action:
    PASS = 0
    SINGLE = 1
    PAIR = 2
    TRIPLE = 3
    FIVE = 4


class HandType(Enum):
    """Hand type constants for last played hand comparisons."""

    SINGLE = "S1"  # Single card
    PAIR = "S2"  # Pair (2 cards)
    TRIPLE = "S3"  # Triple (3 cards)
    # Note: Five-card hands use LastFive class instead


# 5-card category order (Big Two hierarchy)
class FiveCardCategory:
    STRAIGHT = 0
    FLUSH = 1
    FULL_HOUSE = 2
    FOUR_KIND = 3
    STRAIGHT_FLUSH = 4


# 8 straight windows (no 2 in straights) — starts at 3..10 → indices 0..7
STRAIGHT_WINDOWS: list[int] = []
for start in range(8):  # 0..7 represent ranks 3..10
    mask = 0
    for dr in range(5):
        mask |= 1 << (start + dr)
    STRAIGHT_WINDOWS.append(mask)


# ============================================================
# Action space constants
# ============================================================
OFF_PASS = 0
OFF_1 = 1
OFF_2 = OFF_1 + 13  # +33
OFF_3 = OFF_2 + 33  # +31
OFF_5 = OFF_3 + 31  # +1287
N_ACTIONS = OFF_5 + 1287


# ============================================================
# Core Data Classes and Protocols
# ============================================================


@dataclass(frozen=True)
class LastFive:
    category: int
    key: tuple  # comparison key within category


# Import Hand from the canonical location
from ..game.types import Hand


class FiveCardEngine(Protocol):
    def generate(self, hand: Hand, last: LastFive | None) -> list[tuple[int, int, int, int, int]]:
        """Return 5-tuples of slot indices (i<j<k<l<m) that are legal and beat `last` (if provided)."""
        ...
