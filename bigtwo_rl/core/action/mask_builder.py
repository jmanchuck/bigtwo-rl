"""Action mask builder for Big Two game logic."""

from __future__ import annotations
from typing import List, Tuple, Optional

from ..game import Hand, LastFive, FiveCardEngine, HandType, compute_key_and_hand_type
from .constants import OFF_PASS, OFF_1, OFF_5
from .lookups import PAIR_LUT, TRIPLE_LUT, PAIR_ID, TRIPLE_ID
from .utils import comb_index_5


class ActionMaskBuilder:
    """Builds action masks for valid Big Two plays."""

    def __init__(self, five_engine: FiveCardEngine):
        self.five = five_engine

    def single_and_multiples(
        self, hand: Hand, last_kind: Optional[HandType] = None, last_key: Optional[Tuple] = None
    ) -> List[int]:
        """Return action ids for singles/pairs/triples. last_kind in {None, HandType.SINGLE, HandType.PAIR, HandType.TRIPLE}.
        Singles compare by (rank, suit) if last_kind==HandType.SINGLE. Pairs/Triples by rank.
        """
        hand.build_derived()
        ids: List[int] = []

        if last_kind == HandType.SINGLE or last_kind is None:
            ids.extend(self._generate_singles(hand, last_kind, last_key))

        if last_kind == HandType.PAIR or last_kind is None:
            ids.extend(self._generate_pairs(hand, last_kind, last_key))

        if last_kind == HandType.TRIPLE or last_kind is None:
            ids.extend(self._generate_triples(hand, last_kind, last_key))

        return ids

    def _generate_singles(self, hand: Hand, last_kind: Optional[HandType], last_key: Optional[Tuple]) -> List[int]:
        """Generate action ids for single cards."""
        ids: List[int] = []
        played = hand.played
        card = hand.card

        # Singles (13) - optimized with direct array access
        for i in range(13):
            if played[i]:
                continue
            c = card[i]
            r, s = c >> 2, c & 3  # Inline rank_of/suit_of for speed
            if last_kind == HandType.SINGLE and last_key is not None and not ((r, s) > last_key):
                continue
            ids.append(OFF_1 + i)

        return ids

    def _generate_pairs(self, hand: Hand, last_kind: Optional[HandType], last_key: Optional[Tuple]) -> List[int]:
        """Generate action ids for pairs."""
        ids: List[int] = []
        played = hand.played
        card = hand.card

        # Pairs (33) - optimized with early exit conditions
        for i, j in PAIR_LUT:
            if played[i] | played[j]:  # Bitwise OR is faster than logical OR for 0/1
                continue
            ci, cj = card[i], card[j]
            if (ci >> 2) != (cj >> 2):  # Compare ranks directly
                continue
            r = ci >> 2
            if last_kind == HandType.PAIR:
                # For pairs: compare (rank, highest_suit)
                si, sj = ci & 3, cj & 3
                pair_key = (r, max(si, sj))
                if last_key is not None and not (pair_key > last_key):
                    continue
            ids.append(PAIR_ID[(i, j)])

        return ids

    def _generate_triples(self, hand: Hand, last_kind: Optional[HandType], last_key: Optional[Tuple]) -> List[int]:
        """Generate action ids for triples."""
        ids: List[int] = []
        played = hand.played
        card = hand.card

        # Triples (31) - similar optimizations
        for i, j, k in TRIPLE_LUT:
            if played[i] | played[j] | played[k]:
                continue
            ci, cj, ck = card[i], card[j], card[k]
            r0 = ci >> 2
            if r0 != (cj >> 2) or r0 != (ck >> 2):
                continue
            if last_kind == HandType.TRIPLE and last_key is not None and not (r0 > last_key[0]):
                continue
            ids.append(TRIPLE_ID[(i, j, k)])

        return ids

    def five_cards(self, hand: Hand, last: LastFive) -> List[int]:
        """Generate action ids for five-card hands."""
        ids: List[int] = []
        for i, j, k, ell, m in self.five.generate(hand, last):
            ids.append(OFF_5 + comb_index_5(i, j, k, ell, m))
        return ids

    def full_mask_indices(
        self,
        hand: Hand,
        last_played_cards: List[int],  # these are encoded cards, pass in the last non pass cards
        pass_allowed: bool = True,
        is_first_play: bool = False,
        has_control: bool = False,
    ) -> List[int]:
        """
        Returns a list of action indices that are valid for to play. These indices are used to index the length 1365 action space vector.
        """
        ids = []
        if pass_allowed:
            ids.append(OFF_PASS)

        # Handle first play requirement: must contain 3 of diamonds
        if is_first_play:
            ids += self._first_play_mask(hand)
        elif has_control:
            # hasControl means player can play ANY valid combination
            ids += self.single_and_multiples(hand, None, None)
            ids += self.five_cards(hand, None)
        else:
            # Normal play with last hand constraints
            previous_hand_key, previous_hand_type = compute_key_and_hand_type(last_played_cards)
            ids += self.single_and_multiples(hand, previous_hand_type, previous_hand_key)

            if previous_hand_type.is_five_card():
                ids += self.five_cards(hand, LastFive(previous_hand_type, previous_hand_key))

        return sorted(set(ids))

    def _first_play_mask(self, hand: Hand) -> List[int]:
        """Return action ids for first play - must contain 3 of diamonds (card code 0).
        The 3 of diamonds has rank=0, suit=0, so card code = 0<<2|0 = 0.
        """
        hand.build_derived()
        ids: List[int] = []

        # Find the slot containing 3 of diamonds (it'll always be first card when sorted)
        three_diamonds_slot = 0

        # Pre-compute played lookup for faster access
        played = hand.played
        card = hand.card

        # Singles: Only 3D is allowed
        ids.append(OFF_1 + three_diamonds_slot)

        # Pairs: Must include 3D
        for i, j in PAIR_LUT:
            if played[i] or played[j]:
                continue
            if i != three_diamonds_slot and j != three_diamonds_slot:
                continue  # Must include 3D
            ci, cj = card[i], card[j]
            if (ci >> 2) != (cj >> 2):  # Different ranks
                continue
            ids.append(PAIR_ID[(i, j)])

        # Triples: Must include 3D
        for i, j, k in TRIPLE_LUT:
            if played[i] or played[j] or played[k]:
                continue
            if three_diamonds_slot not in (i, j, k):
                continue  # Must include 3D
            ci, cj, ck = card[i], card[j], card[k]
            r0 = ci >> 2
            if r0 != (cj >> 2) or r0 != (ck >> 2):
                continue
            ids.append(TRIPLE_ID[(i, j, k)])

        # Five cards: Must include 3D
        for i, j, k, ell, m in self.five.generate(hand, None):
            if three_diamonds_slot not in (i, j, k, ell, m):
                continue  # Must include 3D
            ids.append(OFF_5 + comb_index_5(i, j, k, ell, m))

        return ids
