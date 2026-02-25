"""Regression tests for focused RL training fixes."""

import numpy as np
import torch as th
from gymnasium import spaces

from bigtwo_rl.core.bigtwo_wrapper import BigTwoWrapper
from bigtwo_rl.core.cards import string_to_card
from bigtwo_rl.training.multi_player_buffer_enhanced import MultiPlayerRolloutBuffer
from bigtwo_rl.training.multi_player_ppo import MultiPlayerPPO
from bigtwo_rl.training.rewards.base_reward import BaseReward


class _ZeroReward(BaseReward):
    def game_reward(self, winner_player: int, player_idx: int, cards_left: int, all_cards_left=None) -> float:
        return 0.0

    def episode_bonus(self, games_won: int, total_games: int, avg_cards_left: float) -> float:
        return 0.0


def _override_hands(env: BigTwoWrapper, hands_strings: list[list[str]], current_player: int) -> None:
    hands = np.zeros((4, 52), dtype=bool)
    for p in range(4):
        for cs in hands_strings[p]:
            hands[p, string_to_card(cs)] = True
    env.game.hands = hands
    env.game.current_player = current_player
    env.game.done = False


def test_midgame_control_is_not_treated_as_first_play() -> None:
    env = BigTwoWrapper(reward_function=_ZeroReward(), games_per_episode=1)
    env.reset(seed=0)

    # Not a game start (card counts are not all 13), but no active last_play => control.
    hands = [
        ["3D", "6C"],
        ["4S", "7H", "8S"],
        ["3C", "9S"],
        ["4D", "TC"],
    ]
    _override_hands(env, hands, current_player=1)  # player 1 does not hold 3♦
    env.game.last_play = None
    env.game.passes_in_row = 0

    mask = env.get_action_mask()
    assert mask.sum() > 0
    assert bool(mask[0]) is True  # pass should be available when player has control


def test_wrapper_rejects_basic_observation_mode() -> None:
    try:
        BigTwoWrapper(reward_function=_ZeroReward(), games_per_episode=1, observation_mode="basic")
    except ValueError as e:
        assert "enhanced" in str(e)
    else:
        raise AssertionError("Expected ValueError for observation_mode='basic'")


def test_enhanced_observation_midgame_control_flags_are_consistent() -> None:
    env = BigTwoWrapper(
        reward_function=_ZeroReward(),
        games_per_episode=1,
        observation_mode="enhanced",
        track_move_history=True,
    )
    env.reset(seed=0)

    hands = [
        ["3D", "6C"],
        ["4S", "7H", "8S"],
        ["3C", "9S"],
        ["4D", "TC"],
    ]
    _override_hands(env, hands, current_player=1)
    env.game.last_play = None
    env.game.passes_in_row = 0

    obs = env._get_observation(1)
    # Previous-hand/control block starts at 383.
    # Flags at 400..402 are [has_control, is_first_play, can_pass].
    assert bool(obs[400]) is True
    assert bool(obs[401]) is False
    assert bool(obs[402]) is True


def test_enhanced_high_card_bits_update_after_real_play() -> None:
    env = BigTwoWrapper(
        reward_function=_ZeroReward(),
        games_per_episode=1,
        observation_mode="enhanced",
        track_move_history=True,
    )
    env.reset(seed=0)

    # Force a controlled setup where player 0 has A♦ and can play without ending game.
    hands = [
        ["AD", "4D", "5D"],
        ["3C", "6C", "7C"],
        ["8C", "9C", "TC"],
        ["JC", "QC", "KC"],
    ]
    _override_hands(env, hands, current_player=0)
    env.game.last_play = None
    env.game.passes_in_row = 0

    pre = env._get_observation(0)
    assert pre.shape[0] == 412
    assert np.all(pre[367:383] == 0.0)

    # Pick a legal non-pass action that includes A or 2.
    mask = env.get_action_mask()
    legal_non_pass = [i for i in np.where(mask)[0] if i != 0]
    assert legal_non_pass

    from bigtwo_rl.core.action import action_to_tuple

    p0_hand = env.game.get_player_hand(0)
    chosen = None
    for aid in legal_non_pass:
        slots = list(action_to_tuple(int(aid)))
        cards = [p0_hand.card[s] for s in slots if 0 <= s < len(p0_hand.card) and not p0_hand.played[s]]
        if any((c // 4) in (11, 12) for c in cards):
            chosen = int(aid)
            break
    assert chosen is not None

    env.step(chosen)
    post = env.current_obs
    # Global high-card record block should now include at least one played high card.
    assert np.sum(post[367:383]) >= 1.0


def test_illegal_action_is_penalized_without_state_mutation() -> None:
    env = BigTwoWrapper(reward_function=_ZeroReward(), games_per_episode=1)
    env.reset(seed=1)
    mask = env.get_action_mask()
    illegal = next(i for i in range(env.action_space.n) if not mask[i])

    pre_hands = env.game.hands.copy()
    pre_player = env.game.current_player
    pre_last_play = env.game.last_play

    _, reward, done, truncated, info = env.step(illegal)

    assert reward == env.invalid_action_penalty
    assert done is False
    assert truncated is False
    assert info["action_was_legal"] is False
    assert info["invalid_action"] is True
    assert info["player_who_moved"] == pre_player
    assert env.game.current_player == pre_player
    assert np.array_equal(env.game.hands, pre_hands)
    if pre_last_play is None:
        assert env.game.last_play is None
    else:
        assert env.game.last_play is not None
        assert np.array_equal(env.game.last_play[0], pre_last_play[0])
        assert env.game.last_play[1] == pre_last_play[1]


def test_game_end_emits_final_rewards_in_info() -> None:
    env = BigTwoWrapper(reward_function=_ZeroReward(), games_per_episode=1)
    env.reset(seed=0)

    # Player 0 can win immediately with a single card.
    hands = [
        ["4D"],
        ["5C", "6C"],
        ["7C", "8C"],
        ["9C", "TC"],
    ]
    _override_hands(env, hands, current_player=0)
    env.game.last_play = None
    env.game.passes_in_row = 0

    mask = env.get_action_mask()
    legal_non_pass = [i for i in np.where(mask)[0] if i != 0]
    assert legal_non_pass, "Expected at least one non-pass legal action"

    _, _, done, _, info = env.step(int(legal_non_pass[0]))
    assert done is True
    assert "final_rewards" in info
    assert len(info["final_rewards"]) == 4


def test_multi_player_buffer_assigns_terminal_rewards_to_last_four_turns() -> None:
    buffer = MultiPlayerRolloutBuffer(
        buffer_size=8,
        observation_space=spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
        action_space=spaces.Discrete(3),
        n_envs=1,
        device="cpu",
    )

    for player in [0, 1, 2, 3]:
        buffer.add(
            obs=np.array([[0.0, 0.0]], dtype=np.float32),
            action=np.array([0]),
            reward=np.array([0.0], dtype=np.float32),
            episode_start=np.array([False]),
            value=th.tensor([0.0]),
            log_prob=th.tensor([0.0]),
            current_player=np.array([player]),
        )

    buffer.assign_final_game_rewards([1.0, -1.0, -2.0, -3.0], env_idx=0)

    # Buffer positions 0..3 correspond to players 0..3 respectively.
    assert np.isclose(buffer.rewards[0], 1.0)
    assert np.isclose(buffer.rewards[1], -1.0)
    assert np.isclose(buffer.rewards[2], -2.0)
    assert np.isclose(buffer.rewards[3], -3.0)
    assert buffer.games_completed == 1


def test_multi_player_buffer_terminal_rewards_are_env_local() -> None:
    buffer = MultiPlayerRolloutBuffer(
        buffer_size=8,
        observation_space=spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
        action_space=spaces.Discrete(3),
        n_envs=2,
        device="cpu",
    )

    # Two timesteps with different acting players across envs.
    buffer.add(
        obs=np.zeros((2, 2), dtype=np.float32),
        action=np.array([0, 0]),
        reward=np.array([0.0, 0.0], dtype=np.float32),
        episode_start=np.array([False, False]),
        value=th.tensor([0.0, 0.0]),
        log_prob=th.tensor([0.0, 0.0]),
        current_player=np.array([0, 2]),
        action_masks=np.ones((2, 3), dtype=bool),
    )
    buffer.add(
        obs=np.zeros((2, 2), dtype=np.float32),
        action=np.array([0, 0]),
        reward=np.array([0.0, 0.0], dtype=np.float32),
        episode_start=np.array([False, False]),
        value=th.tensor([0.0, 0.0]),
        log_prob=th.tensor([0.0, 0.0]),
        current_player=np.array([1, 3]),
        action_masks=np.ones((2, 3), dtype=bool),
    )

    # Assign terminal rewards only for env 1.
    buffer.assign_final_game_rewards([10.0, 11.0, 12.0, 13.0], env_idx=1)

    # Env 0 should remain untouched.
    assert np.allclose(buffer.rewards[:, 0], 0.0)

    # Env 1 should map rewards based on env-1 acting players at those timesteps:
    # pos 1 -> player 3, pos 0 -> player 2.
    assert np.isclose(buffer.rewards[1, 1], 13.0)
    assert np.isclose(buffer.rewards[0, 1], 12.0)


def test_multi_player_buffer_carryover_roundtrip() -> None:
    buffer = MultiPlayerRolloutBuffer(
        buffer_size=8,
        observation_space=spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
        action_space=spaces.Discrete(3),
        n_envs=2,
        device="cpu",
    )

    # Fill six steps so carryover returns the last four.
    for step_idx in range(6):
        cp = np.array([step_idx % 4, (step_idx + 1) % 4], dtype=int)
        obs = np.array([[step_idx, 1.0], [step_idx, 2.0]], dtype=np.float32)
        buffer.add(
            obs=obs,
            action=np.array([0, 1]),
            reward=np.array([0.0, 0.0], dtype=np.float32),
            episode_start=np.array([False, False]),
            value=th.tensor([0.0, 0.0]),
            log_prob=th.tensor([0.0, 0.0]),
            current_player=cp,
            action_masks=np.ones((2, 3), dtype=bool),
        )

    carry = buffer.get_carryover(4)
    assert carry is not None
    assert carry["observations"].shape[0] == 4
    assert np.all(carry["player_who_moved"][0] == np.array([2, 3]))
    assert np.all(carry["player_who_moved"][-1] == np.array([1, 2]))

    buffer.reset()
    primed = buffer.prime_with_carryover(carry)
    assert primed == 4
    assert buffer.pos == 4
    assert np.all(buffer.player_who_moved[0] == np.array([2, 3]))
    assert np.all(buffer.player_who_moved[3] == np.array([1, 2]))


def test_terminal_reward_assignment_uses_carryover_boundary_turns() -> None:
    buffer = MultiPlayerRolloutBuffer(
        buffer_size=8,
        observation_space=spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
        action_space=spaces.Discrete(3),
        n_envs=1,
        device="cpu",
    )

    # Previous rollout tail: players 0,1,2,3
    for player in [0, 1, 2, 3]:
        buffer.add(
            obs=np.array([[0.0, 0.0]], dtype=np.float32),
            action=np.array([0]),
            reward=np.array([0.0], dtype=np.float32),
            episode_start=np.array([False]),
            value=th.tensor([0.0]),
            log_prob=th.tensor([0.0]),
            current_player=np.array([player]),
            action_masks=np.ones((1, 3), dtype=bool),
        )

    carry = buffer.get_carryover(4)
    assert carry is not None
    buffer.reset()
    buffer.prime_with_carryover(carry)

    # First fresh step in new rollout by player 0.
    buffer.add(
        obs=np.array([[1.0, 0.0]], dtype=np.float32),
        action=np.array([1]),
        reward=np.array([0.0], dtype=np.float32),
        episode_start=np.array([False]),
        value=th.tensor([0.0]),
        log_prob=th.tensor([0.0]),
        current_player=np.array([0]),
        action_masks=np.ones((1, 3), dtype=bool),
    )

    # Terminal rewards should map over [new, carry-3, carry-2, carry-1].
    buffer.assign_final_game_rewards([10.0, 11.0, 12.0, 13.0], env_idx=0)
    assert np.isclose(buffer.rewards[4, 0], 10.0)  # player 0 (new step)
    assert np.isclose(buffer.rewards[3, 0], 13.0)  # player 3
    assert np.isclose(buffer.rewards[2, 0], 12.0)  # player 2
    assert np.isclose(buffer.rewards[1, 0], 11.0)  # player 1


def test_multi_player_gae_is_computed_per_env_stream() -> None:
    buffer = MultiPlayerRolloutBuffer(
        buffer_size=4,
        observation_space=spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
        action_space=spaces.Discrete(3),
        n_envs=2,
        device="cpu",
    )

    # Explicitly craft two independent env streams:
    # player 0 acts on steps 0,2 and player 1 acts on steps 1,3 in each env.
    buffer.player_who_moved = np.array(
        [
            [0, 0],
            [1, 1],
            [0, 0],
            [1, 1],
        ],
        dtype=int,
    )
    buffer.rewards = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ],
        dtype=np.float32,
    )
    buffer.values = np.zeros((4, 2), dtype=np.float32)
    buffer.episode_starts = np.zeros((4, 2), dtype=np.float32)

    buffer.compute_multi_player_gae(gamma=1.0, gae_lambda=1.0)

    # Env 0: player-0 stream [0,2] => adv[2]=3, adv[0]=1+3=4
    #        player-1 stream [1,3] => adv[3]=4, adv[1]=2+4=6
    assert np.isclose(buffer.advantages[2, 0], 3.0)
    assert np.isclose(buffer.advantages[0, 0], 4.0)
    assert np.isclose(buffer.advantages[3, 0], 4.0)
    assert np.isclose(buffer.advantages[1, 0], 6.0)

    # Env 1 is computed independently with its own rewards.
    assert np.isclose(buffer.advantages[2, 1], 30.0)
    assert np.isclose(buffer.advantages[0, 1], 40.0)
    assert np.isclose(buffer.advantages[3, 1], 40.0)
    assert np.isclose(buffer.advantages[1, 1], 60.0)

    # Returns should remain value + advantage.
    assert np.allclose(buffer.returns, buffer.advantages)


def test_action_mask_sanitizer_fixes_all_false_rows() -> None:
    masks = np.array(
        [
            [False, False, False, False],
            [True, False, True, False],
        ],
        dtype=bool,
    )
    fixed = MultiPlayerPPO._sanitize_action_masks(masks)
    assert fixed.shape == masks.shape
    assert bool(fixed[0, 0]) is True
    assert fixed[1].sum() == 2


def test_action_mask_sanitizer_handles_1d_input() -> None:
    masks = np.array([False, False, False], dtype=bool)
    fixed = MultiPlayerPPO._sanitize_action_masks(masks)
    assert fixed.shape == (1, 3)
    assert bool(fixed[0, 0]) is True
