"""Big Two action space functionality."""

from .constants import OFF_PASS, OFF_1, OFF_2, OFF_3, OFF_5, N_ACTIONS
from .engines import BitsetFiveCardEngine
from .lookups import (
    PAIR_LUT,
    TRIPLE_LUT,
    PAIR_ID,
    TRIPLE_ID,
    SINGLE_REVERSE,
    PAIR_REVERSE,
    TRIPLE_REVERSE,
    action_to_tuple,
)
from .mask_builder import ActionMaskBuilder
from .utils import comb_index_5, comb_5_from_index, classify_five, _choose_k, _next_comb

__all__ = [
    # Constants
    "OFF_PASS",
    "OFF_1",
    "OFF_2",
    "OFF_3",
    "OFF_5",
    "N_ACTIONS",
    # Engines
    "BitsetFiveCardEngine",
    # Lookup tables
    "PAIR_LUT",
    "TRIPLE_LUT",
    "PAIR_ID",
    "TRIPLE_ID",
    "SINGLE_REVERSE",
    "PAIR_REVERSE",
    "TRIPLE_REVERSE",
    "action_to_tuple",
    # Main functionality
    "ActionMaskBuilder",
    # Utilities
    "comb_index_5",
    "comb_5_from_index",
    "classify_five",
    "_choose_k",
    "_next_comb",
]
