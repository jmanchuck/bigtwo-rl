from __future__ import annotations

from math import comb

from bigtwo_rl.core.action.util import (
    OFF_1,
    OFF_2,
    OFF_3,
    OFF_5,
    OFF_PASS,
    RANKS,
    STRAIGHT_WINDOWS,
    SUITS,
    FiveCardCategory,
    FiveCardEngine,
    Hand,
    HandType,
    LastFive,
    rank_of,
    suit_of,
)


# ============================================================
# Bitset-based, no heavy precompute 5-card engine
# ============================================================
class BitsetFiveCardEngine:
    """Category-pruned generators using only counters & bitsets.
    If `last` is provided, only emit 5-card hands that beat it (category >= last.category and key > last.key in same category).
    """

    def generate(self, hand: Hand, last: LastFive | None) -> list[tuple[int, int, int, int, int]]:
        hand.build_derived()
        out: list[tuple[int, int, int, int, int]] = []
        min_cat = last.category if last else FiveCardCategory.STRAIGHT

        # Higher categories first for quick coverage
        if min_cat <= FiveCardCategory.STRAIGHT_FLUSH:
            out.extend(
                self._gen_straight_flush(
                    hand,
                    last if last and last.category == FiveCardCategory.STRAIGHT_FLUSH else None,
                ),
            )
        if min_cat <= FiveCardCategory.FOUR_KIND:
            out.extend(
                self._gen_four_kind(hand, last if last and last.category == FiveCardCategory.FOUR_KIND else None),
            )
        if min_cat <= FiveCardCategory.FULL_HOUSE:
            out.extend(
                self._gen_full_house(hand, last if last and last.category == FiveCardCategory.FULL_HOUSE else None),
            )
        if min_cat <= FiveCardCategory.FLUSH:
            out.extend(self._gen_flush(hand, last if last and last.category == FiveCardCategory.FLUSH else None))
        if min_cat <= FiveCardCategory.STRAIGHT:
            out.extend(self._gen_straight(hand, last if last and last.category == FiveCardCategory.STRAIGHT else None))
        return out

    # ---------- Category generators ----------
    def _gen_straight_flush(self, hand: Hand, last_same: LastFive | None):
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

    def _gen_four_kind(self, hand: Hand, last_same: LastFive | None):
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

    def _gen_full_house(self, hand: Hand, last_same: LastFive | None):
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

    def _gen_flush(self, hand: Hand, last_same: LastFive | None):
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

    def _gen_straight(self, hand: Hand, last_same: LastFive | None):
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


# ============================================================
# Combinatorics helpers
# ============================================================


def _choose_k(items: list[int], k: int):
    n = len(items)
    if k == 0:
        yield []
        return
    if k > n:
        return
    idx = list(range(k))
    while True:
        yield [items[i] for i in idx]
        # next
        for t in range(k - 1, -1, -1):
            if idx[t] != t + n - k:
                break
        else:
            return
        idx[t] += 1
        for u in range(t + 1, k):
            idx[u] = idx[u - 1] + 1


def _next_comb(idx: list[int], k: int, n: int) -> list[int] | None:
    """Return next k-combination index list in lex order for universe size n, or None if at end."""
    res = idx[:]  # copy
    for t in range(k - 1, -1, -1):
        if res[t] != t + n - k:
            res[t] += 1
            for u in range(t + 1, k):
                res[u] = res[u - 1] + 1
            return res
    return None


# ============================================================
# Action space lookup tables
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


def comb_index_5(i: int, j: int, k: int, ell: int, m: int) -> int:
    arr = [i, j, k, ell, m]
    assert 0 <= i < j < k < ell < m < 13
    res = 0
    start = 0
    for pos, x in enumerate(arr):
        for t in range(start, x):
            res += comb(13 - 1 - t, 5 - 1 - pos)
        start = x + 1
    return res


# ============================================================
# Action mask builder functionality moved to mask_builder.py
# ============================================================


# ============================================================
# Five-card classification (for tests)
# ============================================================


def classify_five(card: list[int], slots: tuple[int, int, int, int, int]) -> tuple[int, tuple]:
    rs = [rank_of(card[i]) for i in slots]
    ss = [suit_of(card[i]) for i in slots]
    cnt = {}
    for r in rs:
        cnt[r] = cnt.get(r, 0) + 1
    is_flush = len(set(ss)) == 1
    bits = 0
    for r in rs:
        bits |= 1 << r
    is_straight = False
    top = -1
    for w_idx, W in enumerate(STRAIGHT_WINDOWS):
        if (bits & W) == W:
            is_straight = True
            top = w_idx + 4
    if is_flush and is_straight:
        return FiveCardCategory.STRAIGHT_FLUSH, (top, ss[0])
    if 4 in cnt.values():
        ra = max((r for r, c in cnt.items() if c == 4))
        kicker = max((r for r, c in cnt.items() if c == 1))
        return FiveCardCategory.FOUR_KIND, (ra, kicker)
    if sorted(cnt.values()) == [2, 3]:
        ra = max((r for r, c in cnt.items() if c == 3))
        rb = max((r for r, c in cnt.items() if c == 2))
        return FiveCardCategory.FULL_HOUSE, (ra, rb)
    if is_flush:
        return FiveCardCategory.FLUSH, tuple(sorted(rs, reverse=True))
    if is_straight:
        return FiveCardCategory.STRAIGHT, (top,)
    raise ValueError("not a five-card hand")
