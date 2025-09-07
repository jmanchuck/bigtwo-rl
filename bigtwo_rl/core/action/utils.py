"""Combinatorial utilities and hand classification for Big Two action space."""

from __future__ import annotations

from math import comb

from bigtwo_rl.core.cards import rank_of, suit_of
from bigtwo_rl.core.game import STRAIGHT_WINDOWS, HandType


def _choose_k(items: list[int], k: int):
    """Generate all k-element combinations of items."""
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


def comb_index_5(i: int, j: int, k: int, ell: int, m: int) -> int:
    """Convert 5-tuple of card slots to combinatorial index."""
    arr = [i, j, k, ell, m]
    assert 0 <= i < j < k < ell < m < 13
    res = 0
    start = 0
    for pos, x in enumerate(arr):
        for t in range(start, x):
            res += comb(13 - 1 - t, 5 - 1 - pos)
        start = x + 1
    return res


def comb_5_from_index(index: int) -> tuple[int, int, int, int, int]:
    """Convert combinatorial index back to 5-tuple of card slots."""
    assert 0 <= index < comb(13, 5), f"Index {index} out of range"

    # Reconstruct the 5-combination from its combinatorial index
    arr = []
    remaining_index = index
    n_remaining = 13
    k_remaining = 5
    start = 0

    for pos in range(5):
        for candidate in range(start, 13):
            count = comb(n_remaining - 1, k_remaining - 1)
            if remaining_index < count:
                arr.append(candidate)
                start = candidate + 1
                n_remaining = 13 - candidate - 1
                k_remaining -= 1
                break
            remaining_index -= count
            n_remaining -= 1

    return tuple(arr)


def classify_five(card: list[int], slots: tuple[int, int, int, int, int]) -> tuple[int, tuple]:
    """Classify a five-card hand and return its type and key."""
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
        return HandType.STRAIGHT_FLUSH, (top, ss[0])
    if 4 in cnt.values():
        ra = max((r for r, c in cnt.items() if c == 4))
        kicker = max((r for r, c in cnt.items() if c == 1))
        return HandType.FOUR_KIND, (ra, kicker)
    if sorted(cnt.values()) == [2, 3]:
        ra = max((r for r, c in cnt.items() if c == 3))
        rb = max((r for r, c in cnt.items() if c == 2))
        return HandType.FULL_HOUSE, (ra, rb)
    if is_flush:
        return HandType.FLUSH, tuple(sorted(rs, reverse=True))
    if is_straight:
        return HandType.STRAIGHT, (top,)
    raise ValueError("not a five-card hand")
