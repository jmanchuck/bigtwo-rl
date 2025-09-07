"""Game-specific types and data structures for Big Two."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Protocol
from enum import Enum

from ..cards import rank_of, suit_of


class Action:
    """Game action constants."""

    PASS = 0
    SINGLE = 1
    PAIR = 2
    TRIPLE = 3
    FIVE = 4


class HandType(Enum):
    """Hand type constants for played hand comparisons."""

    SINGLE = 0  # Single card
    PAIR = 1  # Pair (2 cards)
    TRIPLE = 2  # Triple (3 cards)
    STRAIGHT = 3  # Straight (5 cards)
    FLUSH = 4  # Flush (5 cards)
    FULL_HOUSE = 5  # Full house (5 cards)
    FOUR_KIND = 6  # Four of a kind (5 cards)
    STRAIGHT_FLUSH = 7  # Straight flush (5 cards)

    def is_five_card(self) -> bool:
        """Check if this hand type requires exactly 5 cards."""
        return self in [self.STRAIGHT, self.FLUSH, self.FULL_HOUSE, self.FOUR_KIND, self.STRAIGHT_FLUSH]

    def __lt__(self, other: HandType) -> bool:
        return self.value < other.value

    def __eq__(self, other) -> bool:
        if other is None or not isinstance(other, HandType):
            return False
        return self.value == other.value

    def __le__(self, other: HandType) -> bool:
        return self.value <= other.value

    def __hash__(self) -> int:
        return self.value


@dataclass(frozen=True)
class LastFive:
    """Represents the last played five-card hand for comparison."""

    hand_type: HandType
    key: Tuple  # comparison key within category


@dataclass
class Hand:
    """Fixed 13-slot hand representation.

    Slots never move; we only flip played bits.
    card: length-13 list of uint8 codes (rank<<2|suit)
    played: length-13 list of 0/1 flags
    Derived fields are rebuilt only when needed via build_derived().
    """

    card: List[int]
    played: List[int]

    # Derived fields (rebuilt by build_derived())
    rank_cnt: List[int] = None
    rank_suits_mask: List[int] = None
    suit_cnt: List[int] = None
    suit_rank_bits: List[int] = None
    rank_any_bits: int = 0
    slot_of: List[List[int]] = None  # [rank][suit] -> slot or -1
    _derived_built: bool = False

    def build_derived(self) -> None:
        """Rebuild derived fields from current card/played state."""
        if self._derived_built:
            return
            
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
            
        self._derived_built = True
        
    def invalidate_derived(self) -> None:
        """Mark derived state as needing rebuild after hand changes."""
        self._derived_built = False


class FiveCardEngine(Protocol):
    """Protocol for five-card hand generation engines."""

    def generate(self, hand: Hand, last: Optional[LastFive]) -> List[Tuple[int, int, int, int, int]]:
        """Return 5-tuples of slot indices (i<j<k<l<m) that are legal and beat `last` (if provided)."""
        ...


# Straight windows for game logic (no 2 in straights)
# 8 windows starting at ranks 3..10 → indices 0..7
STRAIGHT_WINDOWS: List[int] = []
for start in range(0, 8):  # 0..7 represent ranks 3..10
    mask = 0
    for dr in range(5):
        mask |= 1 << (start + dr)
    STRAIGHT_WINDOWS.append(mask)
