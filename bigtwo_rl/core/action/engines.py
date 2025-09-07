"""Five-card hand generation engines for Big Two."""

from __future__ import annotations
from typing import List, Tuple, Optional

from core import RANKS, SUITS, rank_of
from game import Hand, LastFive, FiveCardEngine, HandType, STRAIGHT_WINDOWS
from .utils import _choose_k, _next_comb


class BitsetFiveCardEngine:
    """Category-pruned generators using only counters & bitsets.
    If `last` is provided, only emit 5-card hands that beat it (hand_type >= last.hand_type and key > last.key in same category).
    """

    def generate(self, hand: Hand, last: LastFive | None) -> List[Tuple[int, int, int, int, int]]:
        hand.build_derived()
        out: List[Tuple[int, int, int, int, int]] = []

        # If last is None, generate all possible five-card combinations
        if last is None:
            out.extend(self._gen_straight_flush(hand, None))
            out.extend(self._gen_four_kind(hand, None))
            out.extend(self._gen_full_house(hand, None))
            out.extend(self._gen_flush(hand, None))
            out.extend(self._gen_straight(hand, None))
            return out

        min_cat = last.hand_type

        # Higher categories first for quick coverage
        if min_cat <= HandType.STRAIGHT_FLUSH:
            out.extend(self._gen_straight_flush(hand, last if last.hand_type == HandType.STRAIGHT_FLUSH else None))
        if min_cat <= HandType.FOUR_KIND:
            out.extend(self._gen_four_kind(hand, last if last.hand_type == HandType.FOUR_KIND else None))
        if min_cat <= HandType.FULL_HOUSE:
            out.extend(self._gen_full_house(hand, last if last.hand_type == HandType.FULL_HOUSE else None))
        if min_cat <= HandType.FLUSH:
            out.extend(self._gen_flush(hand, last if last.hand_type == HandType.FLUSH else None))
        if min_cat <= HandType.STRAIGHT:
            out.extend(self._gen_straight(hand, last if last.hand_type == HandType.STRAIGHT else None))
        return out

    def _gen_straight_flush(self, hand: Hand, last_same: Optional[LastFive]):
        for s in SUITS:
            bits = hand.suit_rank_bits[s]
            for w_idx, W in enumerate(STRAIGHT_WINDOWS):
                if (bits & W) == W:
                    top = w_idx + 4
                    key = (top, s)  # tie by suit if desired; remove s if not
                    if last_same and not (key > last_same.key):
                        continue
                    ranks = [w_idx + d for d in range(5)]
                    slots = tuple(sorted(hand.slot_of[r][s] for r in ranks))
                    yield slots

    def _gen_four_kind(self, hand: Hand, last_same: Optional[LastFive]):
        for ra in RANKS:
            if hand.rank_cnt[ra] == 4:
                quad_slots = [hand.slot_of[ra][s] for s in SUITS if (hand.rank_suits_mask[ra] >> s) & 1]
                for i, c in enumerate(hand.card):
                    if hand.played[i]:
                        continue
                    if rank_of(c) == ra:
                        continue
                    key = (ra, rank_of(c))
                    if last_same and not (key > last_same.key):
                        continue
                    yield tuple(sorted(quad_slots + [i]))

    def _gen_full_house(self, hand: Hand, last_same: Optional[LastFive]):
        triples = [r for r in RANKS if hand.rank_cnt[r] >= 3]
        pairs = [r for r in RANKS if hand.rank_cnt[r] >= 2]
        for ra in triples:
            for rb in pairs:
                if rb == ra:
                    continue
                key = (ra, rb)
                if last_same and not (key > last_same.key):
                    continue
                ra_suits = [s for s in SUITS if (hand.rank_suits_mask[ra] >> s) & 1]
                rb_suits = [s for s in SUITS if (hand.rank_suits_mask[rb] >> s) & 1]
                ra_slots = [hand.slot_of[ra][s] for s in ra_suits]
                rb_slots = [hand.slot_of[rb][s] for s in rb_suits]
                for A in _choose_k(ra_slots, 3):
                    for B in _choose_k(rb_slots, 2):
                        yield tuple(sorted(A + B))

    def _gen_flush(self, hand: Hand, last_same: Optional[LastFive]):
        for s in SUITS:
            c = hand.suit_cnt[s]
            if c < 5:
                continue
            ranks = [r for r in RANKS if (hand.suit_rank_bits[s] >> r) & 1]  # ascending
            n = len(ranks)
            if n < 5:
                continue
            # iterate combinations; optionally skip losers if last_same provided
            start_idx = [0, 1, 2, 3, 4]
            if last_same:
                # last key is 5 ranks in DESC order
                last_desc = list(last_same.key)
                # convert to ascending
                need = list(reversed(last_desc))
                # map to positions in ranks; if any missing -> the smallest combo already beats
                pos = []
                found_all = True
                for v in need:
                    try:
                        pos.append(ranks.index(v))
                    except ValueError:
                        found_all = False
                        break
                if found_all:
                    # advance to lexicographic successor of pos
                    next_pos = _next_comb(pos, k=5, n=n)
                    if next_pos is None:
                        continue  # nothing can beat
                    start_idx = next_pos
            # iterate all 5-combos from start_idx upwards
            idx = start_idx
            while idx is not None:
                selected_ranks = [ranks[t] for t in idx]
                key = tuple(sorted(selected_ranks, reverse=True))
                if last_same:
                    if not (key > last_same.key):
                        idx = _next_comb(idx, k=5, n=n)
                        continue
                slots = tuple(sorted(hand.slot_of[ranks[t]][s] for t in idx))
                yield slots
                idx = _next_comb(idx, k=5, n=n)

    def _gen_straight(self, hand: Hand, last_same: Optional[LastFive]):
        for w_idx, W in enumerate(STRAIGHT_WINDOWS):
            if (hand.rank_any_bits & W) != W:
                continue
            top = w_idx + 4
            if last_same and not (top > last_same.key[0]):
                continue  # require strictly higher top rank
            ranks = [w_idx + d for d in range(5)]
            suit_lists = [[s for s in SUITS if (hand.rank_suits_mask[r] >> s) & 1] for r in ranks]
            slots = [0] * 5

            # DFS over at most 4^5 choices
            def dfs(d: int):
                if d == 5:
                    yield tuple(sorted(slots))
                    return
                r = ranks[d]
                for s in suit_lists[d]:
                    slots[d] = hand.slot_of[r][s]
                    yield from dfs(d + 1)

            yield from dfs(0)
