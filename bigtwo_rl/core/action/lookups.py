"""Action space lookup tables and reverse mappings for Big Two."""

from __future__ import annotations

from math import comb

from .constants import OFF_1, OFF_2, OFF_3, OFF_5, OFF_PASS
from .utils import comb_5_from_index

# ============================================================
# Forward lookup tables
# ============================================================

# Pairs LUT (33)
PAIR_LUT: list[tuple[int, int]] = []
PAIR_ID: dict[tuple[int, int], int] = {}
for i in range(13):
    for d in (1, 2, 3):
        j = i + d
        if j <= 12:
            PAIR_LUT.append((i, j))
for idx, (i, j) in enumerate(PAIR_LUT):
    PAIR_ID[(i, j)] = OFF_2 + idx

# Triples LUT (31)
TRIPLE_LUT: list[tuple[int, int, int]] = []
TRIPLE_ID: dict[tuple[int, int, int], int] = {}
for i in range(11):
    TRIPLE_LUT.append((i, i + 1, i + 2))  # A
    if i <= 9:
        TRIPLE_LUT.append((i, i + 1, i + 3))  # B
        TRIPLE_LUT.append((i, i + 2, i + 3))  # C
for idx, t in enumerate(TRIPLE_LUT):
    TRIPLE_ID[t] = OFF_3 + idx


# ============================================================
# Reverse lookup tables
# ============================================================

SINGLE_REVERSE: dict[int, tuple[int]] = {OFF_1 + i: (i,) for i in range(13)}
PAIR_REVERSE: dict[int, tuple[int, int]] = {action_id: pair for pair, action_id in PAIR_ID.items()}
TRIPLE_REVERSE: dict[int, tuple[int, int, int]] = {action_id: triple for triple, action_id in TRIPLE_ID.items()}


def action_to_tuple(action_id: int) -> tuple[int, ...]:
    """Convert action ID back to tuple of card slot indices.

    Args:
        action_id: Action index from 0-1364

    Returns:
        Tuple of card slot indices representing the play

    Raises:
        ValueError: If action_id is not in valid range

    """
    if action_id == OFF_PASS:
        return ()  # Pass action
    if OFF_1 <= action_id < OFF_2:
        return SINGLE_REVERSE[action_id]
    if OFF_2 <= action_id < OFF_3:
        return PAIR_REVERSE[action_id]
    if OFF_3 <= action_id < OFF_5:
        return TRIPLE_REVERSE[action_id]
    if OFF_5 <= action_id < OFF_5 + comb(13, 5):
        five_index = action_id - OFF_5
        return comb_5_from_index(five_index)
    raise ValueError(f"Invalid action_id: {action_id}")
