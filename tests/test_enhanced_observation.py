"""Tests for enhanced observation builder and wrapper integration."""

import numpy as np

from bigtwo_rl.core.bigtwo_wrapper import BigTwoWrapper
from bigtwo_rl.core.observation import EnhancedObservationBuilder
from bigtwo_rl.training.rewards.base_reward import BaseReward


class _ZeroReward(BaseReward):
    def game_reward(self, winner_player: int, player_idx: int, cards_left: int, all_cards_left=None) -> float:
        return 0.0

    def episode_bonus(self, games_won: int, total_games: int, avg_cards_left: float) -> float:
        return 0.0


def test_enhanced_builder_size() -> None:
    builder = EnhancedObservationBuilder()
    assert builder.observation_size == 412


def test_wrapper_enhanced_observation_shape() -> None:
    env = BigTwoWrapper(
        reward_function=_ZeroReward(),
        games_per_episode=1,
        track_move_history=True,
        observation_mode="enhanced",
    )
    obs, _ = env.reset(seed=123)
    assert obs.shape == (412,)
    assert env.observation_space.shape == (412,)


def test_enhanced_history_updates_after_move() -> None:
    env = BigTwoWrapper(
        reward_function=_ZeroReward(),
        games_per_episode=1,
        track_move_history=True,
        observation_mode="enhanced",
    )
    obs0, _ = env.reset(seed=5)
    mask = env.get_action_mask()
    action = int(np.where(mask)[0][0])
    env.step(action)
    obs1 = env.current_obs
    # Observation should change after at least one move is recorded.
    assert not np.array_equal(obs0, obs1)


def test_enhanced_global_high_record_bits_from_move_history() -> None:
    builder = EnhancedObservationBuilder()
    hand = np.zeros(13, dtype=np.int32).tolist()
    played = np.ones(13, dtype=np.int32).tolist()
    # Keep one card available so hand is valid.
    hand[0] = 0
    played[0] = 0
    from bigtwo_rl.core.game import Hand

    h = Hand(card=hand, played=played)
    # Simulate moves where A♦ (44) and 2♠ (51) were played.
    m1 = np.zeros(52, dtype=np.int8)
    m1[44] = 1
    m2 = np.zeros(52, dtype=np.int8)
    m2[51] = 1
    obs = builder.build_observation(
        hand=h,
        current_player=0,
        player_card_counts=[13, 13, 13, 13],
        last_played_cards=[],
        passes=0,
        is_first_play=False,
        move_history=[(m1, 1, "single"), (m2, 2, "single")],
        has_control=False,
        can_pass=True,
    )
    # Global high-card record block is 367:383 (Q/K/A/2 of all suits).
    high = obs[367:383]
    assert high.shape[0] == 16
    # A♦ (deck idx 44 => block idx 8), 2♠ (deck idx 51 => block idx 15)
    assert high[8] == 1.0
    assert high[15] == 1.0
