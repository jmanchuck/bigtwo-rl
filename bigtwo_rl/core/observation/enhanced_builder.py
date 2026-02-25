"""Reference-inspired observation builder (412 dims).

Layout (close to reference implementation, adapted for this ruleset):
- 286: current hand slot features (13 slots x 22 features)
- 81: opponent context (3 relative opponents x 27 features)
- 16: global high-card depletion tracker (Q/K/A/2 of all suits)
- 29: previous-hand / control / pass context
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..cards import rank_of, suit_of
from ..game.types import Hand
from .basic_builder import BasicObservationBuilder


class EnhancedObservationBuilder(BasicObservationBuilder):
    """Reference-inspired 412-dim observation vector."""

    def __init__(self):
        super().__init__()
        self.observation_size = 412

    @staticmethod
    def _straight_windows() -> list[set[int]]:
        # Rank encoding: 0..12 maps to 3..A,2; valid straights include JQKA2.
        return [set(range(start, start + 5)) for start in range(0, 9)]

    def _encode_hand_slots(self, hand: Hand) -> np.ndarray:
        """Encode current hand as 13 fixed slots x 22 features = 286."""
        hand.build_derived()
        out = np.zeros((13, 22), dtype=np.float32)
        windows = self._straight_windows()

        for slot in range(13):
            if hand.played[slot]:
                continue
            c = hand.card[slot]
            r = rank_of(c)
            s = suit_of(c)

            # 0..12: rank one-hot
            out[slot, r] = 1.0
            # 13..16: suit one-hot
            out[slot, 13 + s] = 1.0
            # 17..21: combinability/potential flags
            out[slot, 17] = 1.0 if hand.rank_cnt[r] >= 2 else 0.0  # in pair
            out[slot, 18] = 1.0 if hand.rank_cnt[r] >= 3 else 0.0  # in triple
            out[slot, 19] = 1.0 if hand.rank_cnt[r] >= 4 else 0.0  # in four-kind
            out[slot, 20] = 1.0 if any((r in w) and all(hand.rank_cnt[x] > 0 for x in w) for w in windows) else 0.0
            out[slot, 21] = 1.0 if hand.suit_cnt[s] >= 5 else 0.0  # in flush

        return out.reshape(-1)

    @staticmethod
    def _extract_cards_played_by_actor(
        move_history: list[tuple[np.ndarray, int, str]] | None,
        actor_idx: int,
    ) -> np.ndarray:
        """Return 52-bit OR mask of cards played by a specific actor."""
        out = np.zeros(52, dtype=np.float32)
        if not move_history:
            return out
        for cards_mask, actor, _ in move_history:
            if actor != actor_idx:
                continue
            if isinstance(cards_mask, np.ndarray) and cards_mask.shape[0] == 52:
                out = np.maximum(out, cards_mask.astype(np.float32))
        return out

    def _encode_opponent_blocks(
        self,
        current_player: int,
        player_card_counts: list[int],
        last_played_cards: list[int],
        has_control: bool,
        is_first_play: bool,
        move_history: list[tuple[np.ndarray, int, str]] | None,
    ) -> np.ndarray:
        """Encode 3 relative opponent blocks, 27 dims each => 81 dims."""
        out = np.zeros((3, 27), dtype=np.float32)

        need_single = float((not has_control) and (not is_first_play) and len(last_played_cards) == 1)
        need_pair = float((not has_control) and (not is_first_play) and len(last_played_cards) == 2)
        need_triple = float((not has_control) and (not is_first_play) and len(last_played_cards) == 3)
        need_five = float((not has_control) and (not is_first_play) and len(last_played_cards) == 5)
        control_flag = float(has_control)

        for rel in range(3):
            opp = (current_player + rel + 1) % 4
            cnt = int(np.clip(player_card_counts[opp], 0, 13))
            # 0..13: one-hot count bins for opponent remaining cards.
            out[rel, cnt] = 1.0

            # 14..21: A/2 played by this opponent (deck indices 44..51).
            played_mask = self._extract_cards_played_by_actor(move_history, opp)
            out[rel, 14:22] = played_mask[44:52]

            # 22..26: pressure/control context.
            out[rel, 22] = need_single
            out[rel, 23] = need_pair
            out[rel, 24] = need_triple
            out[rel, 25] = need_five
            out[rel, 26] = control_flag

        return out.reshape(-1)

    @staticmethod
    def _analyze_last_hand(last_played_cards: list[int]) -> tuple[int, int, np.ndarray]:
        """Return dominant rank, suit and 9 hand-type flags for previous hand."""
        flags = np.zeros(9, dtype=np.float32)
        if not last_played_cards:
            return -1, -1, flags

        cards = sorted(last_played_cards)
        ranks = np.array([rank_of(c) for c in cards], dtype=np.int32)
        suits = np.array([suit_of(c) for c in cards], dtype=np.int32)
        n = len(cards)

        if n == 1:
            flags[0] = 1.0  # single
            return int(ranks[0]), int(suits[0]), flags
        if n == 2:
            flags[1] = 1.0  # pair
            return int(ranks[0]), int(np.max(suits)), flags
        if n == 3:
            flags[2] = 1.0  # triple
            return int(ranks[0]), int(np.max(suits)), flags
        if n != 5:
            return int(np.max(ranks)), int(np.max(suits)), flags

        uniq_r, cnts = np.unique(ranks, return_counts=True)
        is_flush = len(np.unique(suits)) == 1
        is_straight = np.all(np.diff(np.sort(ranks)) == 1)
        is_full_house = sorted(cnts.tolist()) == [2, 3]
        is_four = np.any(cnts == 4)
        is_straight_flush = bool(is_straight and is_flush)

        if is_straight:
            flags[3] = 1.0
        if is_flush:
            flags[4] = 1.0
        if is_full_house:
            flags[5] = 1.0
        if is_four:
            flags[6] = 1.0
        if is_straight_flush:
            flags[7] = 1.0
        flags[8] = 1.0  # generic five-card marker

        if is_full_house:
            trip_rank = int(uniq_r[np.argmax(cnts)])
            return trip_rank, -1, flags
        if is_four:
            four_rank = int(uniq_r[np.argmax(cnts)])
            return four_rank, -1, flags
        return int(np.max(ranks)), int(suits[np.argmax(ranks)]), flags

    def _encode_global_high_cards(self, move_history: list[tuple[np.ndarray, int, str]] | None) -> np.ndarray:
        """Encode played Q/K/A/2 cards globally (16 bits, deck indices 36..51)."""
        out = np.zeros(16, dtype=np.float32)
        if not move_history:
            return out
        for cards_mask, _, _ in move_history:
            if isinstance(cards_mask, np.ndarray) and cards_mask.shape[0] == 52:
                out = np.maximum(out, cards_mask[36:52].astype(np.float32))
        return out

    def _encode_prev_context(
        self,
        last_played_cards: list[int],
        has_control: bool,
        is_first_play: bool,
        can_pass: bool,
    ) -> np.ndarray:
        """Encode previous-hand/control context into 29 dims."""
        out = np.zeros(29, dtype=np.float32)
        rank, suit, hand_flags = self._analyze_last_hand(last_played_cards)

        # 0..12 rank one-hot
        if 0 <= rank < 13:
            out[rank] = 1.0
        # 13..16 suit one-hot
        if 0 <= suit < 4:
            out[13 + suit] = 1.0

        # 17..28 flags
        out[17] = float(has_control)
        out[18] = float(is_first_play)
        out[19] = float(can_pass)
        # 20..28 last hand type flags:
        # single,pair,triple,straight,flush,full_house,four_kind,straight_flush,five_generic
        out[20:29] = hand_flags
        return out

    def build_observation(
        self,
        hand: Hand,
        current_player: int,
        player_card_counts: list[int],
        last_played_cards: list[int],
        passes: int,
        is_first_play: bool,
        move_history: list[tuple[np.ndarray, int, str]] | None = None,
        has_control: bool = False,
        can_pass: bool = True,
    ) -> np.ndarray:
        """Build complete 412-dim reference-inspired observation vector."""
        hand_slots = self._encode_hand_slots(hand)
        opp = self._encode_opponent_blocks(
            current_player=current_player,
            player_card_counts=player_card_counts,
            last_played_cards=last_played_cards,
            has_control=has_control,
            is_first_play=is_first_play,
            move_history=move_history,
        )
        high_global = self._encode_global_high_cards(move_history)
        prev_ctx = self._encode_prev_context(
            last_played_cards=last_played_cards,
            has_control=has_control,
            is_first_play=is_first_play,
            can_pass=can_pass,
        )
        return np.concatenate([hand_slots, opp, high_global, prev_ctx]).astype(np.float32)

    def get_feature_info(self) -> dict[str, Any]:
        return {
            "hand_slot_features": 286,
            "opponent_context_features": 81,
            "global_high_card_record": 16,
            "previous_hand_context": 29,
            "total_size": 412,
        }
