"""Deterministic action-count tests by overriding game state in the wrapper.

We construct exact hands and assert the number of legal actions from the
BigTwoWrapper equals what the ActionMaskBuilder would produce for the same state.
This validates the integration between the game engine and wrapper mask logic.
"""

import os
import sys

import numpy as np

# Ensure project root on path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigtwo_rl.core.bigtwo_wrapper import BigTwoWrapper
from bigtwo_rl.core.cards import THREE_DIAMONDS, string_to_card
from bigtwo_rl.training.rewards.base_reward import BaseReward


class _ZeroReward(BaseReward):
    def game_reward(self, winner_player: int, player_idx: int, cards_left: int, all_cards_left=None) -> float:
        return 0.0

    def episode_bonus(self, games_won: int, total_games: int, avg_cards_left: float) -> float:
        return 0.0


PLAYER_HANDS = [
    ["3D", "3H", "4H", "5D", "5C", "5H", "5S", "6C", "7S", "8D", "9H", "TH", "KD"],
    ["4S", "7H", "8S", "9C", "TD", "TS", "QH", "QS", "KC", "KH", "AD", "AH", "2D"],
    ["3C", "3S", "4C", "6S", "7D", "8H", "9S", "JD", "JC", "JH", "KS", "2H", "2S"],
    ["4D", "6D", "6H", "7C", "8C", "9D", "TC", "JS", "QD", "QC", "AC", "AS", "2C"],
]


def _make_env() -> BigTwoWrapper:
    env = BigTwoWrapper(reward_function=_ZeroReward(), num_players=4, games_per_episode=1)
    env.reset(seed=0)
    return env


def _override_hands(env: BigTwoWrapper, hands_strings):
    """Override game hands with encoded cards from provided string arrays."""
    hands = np.zeros((4, 52), dtype=bool)
    for p in range(4):
        for cs in hands_strings[p]:
            hands[p, string_to_card(cs)] = True
    env.game.hands = hands

    # Set starting player as the one holding 3♦ to simulate first play
    starters = [p for p in range(4) if env.game.hands[p, THREE_DIAMONDS]]
    env.game.current_player = starters[0] if starters else 0
    env.game.last_play = None
    env.game.passes_in_row = 0
    env.game.done = False


def _play_cards(env: BigTwoWrapper, card_strs):
    """Play exact cards for the current player by finding the matching legal move."""
    legal = env.game.legal_moves(env.game.current_player)
    if not card_strs:  # pass
        # Find pass mask (all False)
        pass_idx = next(i for i, m in enumerate(legal) if not np.any(m))
        env.game.step(pass_idx)
        return

    # Build mask for desired cards
    mask = np.zeros(52, dtype=bool)
    for cs in card_strs:
        mask[string_to_card(cs)] = True

    # Find matching legal move by exact mask equality
    for idx, m in enumerate(legal):
        if np.array_equal(m, mask):
            env.game.step(idx)
            return
    raise AssertionError(f"Desired move {card_strs} not found among legal moves")


def test_first_play_mask_count_matches_builder():
    env = _make_env()
    _override_hands(env, PLAYER_HANDS)

    # Wrapper mask
    mask = env.get_action_mask()

    # Expected ids from builder with first-play constraints
    hand = env.game.get_player_hand(env.game.current_player)
    ids = env.action_masker.full_mask_indices(
        hand=hand,
        last_played_cards=[],
        pass_allowed=False,
        is_first_play=True,
        has_control=False,
    )

    # Numeric expectation from the known hand configuration
    expected_count = 1 + 1 + 4 + 1 + 4  # 3D single, 33 pair, 4 straights to 7, 4K, 4 FH

    assert mask.sum() == len(ids) == expected_count
    assert not mask[0]  # pass not allowed at first play


def test_pairs_progression_counts():
    env = _make_env()
    _override_hands(env, PLAYER_HANDS)

    # P0 plays pair of 3s
    _play_cards(env, ["3D", "3H"])  # now P1's turn

    # P1 mask should allow: pass + beating pairs (T,Q,K,A) -> 5
    mask_p1 = env.get_action_mask()
    assert mask_p1.sum() == 5
    assert bool(mask_p1[0]) is True

    # P1 plays pair of Tens
    _play_cards(env, ["TD", "TS"])  # now P2's turn

    # P2: pass + beating pairs (J variations, 2) -> 5
    mask_p2 = env.get_action_mask()
    assert mask_p2.sum() == 5
    assert bool(mask_p2[0]) is True

    # P2 plays pair of Jacks (choose any two: JD, JC)
    _play_cards(env, ["JD", "JC"])  # now P3's turn

    # P3: pass + beating pairs (Q, A) -> 3
    mask_p3 = env.get_action_mask()
    assert mask_p3.sum() == 3
    assert bool(mask_p3[0]) is True


def test_pass_increments_and_counts():
    env = _make_env()
    _override_hands(env, PLAYER_HANDS)

    # P0 plays pair of 3s
    _play_cards(env, ["3D", "3H"])  # P1

    # P1 passes
    _play_cards(env, [])  # P2
    assert env.game.passes_in_row == 1

    # P2 counts now include: pass + pair 3 + 3 combos of pair J + pair 2 -> 6
    mask_p2 = env.get_action_mask()
    assert mask_p2.sum() == 6
    assert bool(mask_p2[0]) is True


def test_has_control_allows_any_play_counts():
    env = _make_env()
    _override_hands(env, PLAYER_HANDS)

    # Simulate has-control state for current player (bypass strict game transitions)
    # Wrapper's has_control() returns True when passes_in_row >= 3 and not first play
    env.game.passes_in_row = 3
    env.game.last_play = (np.zeros(52, dtype=bool), env.game.current_player)

    # Wrapper mask under has_control
    mask = env.get_action_mask()

    # Expected ids from builder for has_control (any valid combo), no pass
    hand = env.game.get_player_hand(env.game.current_player)
    ids = env.action_masker.full_mask_indices(
        hand=hand,
        last_played_cards=[],
        pass_allowed=True,  # wrapper will allow pass when not first play
        is_first_play=False,
        has_control=True,
    )

    assert mask.sum() == len(ids)
    assert bool(mask[0]) is True
