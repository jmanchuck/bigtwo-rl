"""Hand classification logic for Big Two."""

from typing import List, Tuple

from .types import HandType, STRAIGHT_WINDOWS


def compute_key_and_hand_type(encoded_cards: List[int]) -> Tuple[Tuple, HandType]:
    """
    Compute comparison key and hand type from encoded cards - optimized for speed.

    Returns:
        - Single: (rank, suit)
        - Pair: (rank, highest_suit)
        - Triple: (rank,)
        - Five-card: key from hand classification

    Raises:
        ValueError: For invalid hands
    """
    n = len(encoded_cards)

    if n == 1:
        # Single: inline rank_of/suit_of for speed
        c = encoded_cards[0]
        return (c >> 2, c & 3), HandType.SINGLE

    elif n == 2:
        # Pair: check ranks match, return (rank, highest_suit)
        c1, c2 = encoded_cards
        r1, r2 = c1 >> 2, c2 >> 2
        s1, s2 = c1 & 3, c2 & 3
        if r1 != r2:
            raise ValueError("Pair ranks don't match")
        return (r1, max(s1, s2)), HandType.PAIR

    elif n == 3:
        # Triple: check all ranks match
        c1, c2, c3 = encoded_cards
        r1, r2, r3 = c1 >> 2, c2 >> 2, c3 >> 2
        if r1 != r2 or r1 != r3:
            raise ValueError("Triple ranks don't match")
        return (r1,), HandType.TRIPLE

    elif n == 5:
        # Five-card: fast classification
        return _classify_five_fast(encoded_cards)

    raise ValueError(f"Invalid hand size: {n}")


def _classify_five_fast(cards: List[int]) -> Tuple[Tuple, HandType]:
    """Fast five-card classification avoiding function calls."""
    # Extract ranks and suits using bitwise ops
    rs = [c >> 2 for c in cards]
    ss = [c & 3 for c in cards]

    # Count ranks
    cnt = [0] * 13
    for r in rs:
        cnt[r] += 1

    # Check flush
    is_flush = len(set(ss)) == 1

    # Check straight
    bits = 0
    for r in rs:
        bits |= 1 << r

    is_straight = False
    top = -1
    for w_idx, W in enumerate(STRAIGHT_WINDOWS):
        if (bits & W) == W:
            is_straight = True
            top = w_idx + 4
            break

    # Classify hand type (highest to lowest)
    if is_flush and is_straight:
        return (top, ss[0]), HandType.STRAIGHT_FLUSH

    # Check four of a kind
    if 4 in cnt:
        ra = cnt.index(4)  # Four-of-a-kind rank
        kicker = max(i for i, c in enumerate(cnt) if c == 1)
        return (ra, kicker), HandType.FOUR_KIND

    # Check full house
    if 3 in cnt and 2 in cnt:
        ra = cnt.index(3)  # Triple rank
        rb = cnt.index(2)  # Pair rank
        return (ra, rb), HandType.FULL_HOUSE

    # Check flush
    if is_flush:
        return tuple(sorted(rs, reverse=True)), HandType.FLUSH

    # Check straight
    if is_straight:
        return (top,), HandType.STRAIGHT

    raise ValueError("Invalid five-card hand")


def classify_five(card: List[int], slots: Tuple[int, int, int, int, int]) -> Tuple[HandType, Tuple]:
    """Classify a five-card hand from card codes and slot indices."""
    encoded_cards = [card[i] for i in slots]
    key, hand_type = compute_key_and_hand_type(encoded_cards)
    return hand_type, key
