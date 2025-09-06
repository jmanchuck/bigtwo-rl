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


def rank_of(code):
    return code >> 2


def suit_of(code):
    return code & 3


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


@dataclass
class Hand:
    """Fixed 13-slot hand. Slots never move; we only flip played bits.
    card: length-13 list of uint8 codes (rank<<2|suit)
    played: length-13 list of 0/1 flags
    Derived fields are rebuilt each call via build_derived().
    """

    card: list[int]
    played: list[int]
    # derived
    rank_cnt: list[int] = None
    rank_suits_mask: list[int] = None
    suit_cnt: list[int] = None
    suit_rank_bits: list[int] = None
    rank_any_bits: int = 0
    slot_of: list[list[int]] = None  # [rank][suit] -> slot or -1

    def build_derived(self) -> None:
        self.rank_cnt = [0] * 13
        self.rank_suits_mask = [0] * 13
        self.suit_cnt = [0] * 4
        self.suit_rank_bits = [0] * 4
        self.rank_any_bits = 0
        self.slot_of = [[-1] * 4 for _ in range(13)]
        for i, c in enumerate(self.card):
            if self.played[i]:
                continue
            r, s = rank_of(c), suit_of(c)
            self.rank_cnt[r] += 1
            self.rank_suits_mask[r] |= 1 << s
            self.suit_cnt[s] += 1
            self.suit_rank_bits[s] |= 1 << r
            self.rank_any_bits |= 1 << r
            self.slot_of[r][s] = i


class FiveCardEngine(Protocol):
    def generate(self, hand: Hand, last: LastFive | None) -> list[tuple[int, int, int, int, int]]:
        """Return 5-tuples of slot indices (i<j<k<l<m) that are legal and beat `last` (if provided)."""
        ...
