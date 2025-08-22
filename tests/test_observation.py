#!/usr/bin/env python3
"""Test the new observation space with full previous hand information."""

import numpy as np
from bigtwo_rl.core.bigtwo import ToyBigTwoFullRules
from bigtwo_rl.core.rl_wrapper import BigTwoRLWrapper
from bigtwo_rl.core.observation_builder import standard_observation


def test_observation_space():
    """Test that the new observation space includes full previous hand."""
    print("Testing observation space with full previous hand...")

    # Test raw environment
    env = ToyBigTwoFullRules(num_players=4)
    obs = env.reset()

    # Check observation keys
    expected_keys = {"hand", "last_play", "last_play_exists", "legal_moves"}
    assert set(obs.keys()) == expected_keys, (
        f"Missing keys: {expected_keys - set(obs.keys())}"
    )

    # Check dimensions
    assert obs["hand"].shape == (52,), f"Hand shape: {obs['hand'].shape}"
    assert obs["last_play"].shape == (52,), f"Last play shape: {obs['last_play'].shape}"
    assert obs["last_play_exists"] in [
        0,
        1,
    ], f"Last play exists: {obs['last_play_exists']}"

    print(f"✓ Hand cards: {obs['hand'].sum()}")
    print(f"✓ Last play cards: {obs['last_play'].sum()}")
    print(f"✓ Last play exists: {obs['last_play_exists']}")

    # Make a move and check last play is recorded
    legal_moves = obs["legal_moves"]
    if legal_moves:
        # Play first legal move
        obs, rewards, done, info = env.step(0)
        print(f"✓ After move - last play cards: {obs['last_play'].sum()}")
        print(f"✓ After move - last play exists: {obs['last_play_exists']}")

        # The last play should now contain the cards that were played
        assert obs["last_play_exists"] == 1, "Last play should exist after a move"
        assert obs["last_play"].sum() > 0, "Last play should contain cards"

    print("Raw environment test passed!")


def test_rl_wrapper():
    """Test the RL wrapper with new observation space."""
    print("\nTesting RL wrapper...")

    wrapper = BigTwoRLWrapper(
        observation_config=standard_observation(), num_players=4, games_per_episode=1
    )
    obs, info = wrapper.reset()

    # Check observation shape
    assert obs.shape == (109,), f"Observation shape: {obs.shape}"

    # Check observation structure: hand(52) + last_play(52) + hand_sizes(4) + last_play_exists(1)
    hand_part = obs[:52]
    last_play_part = obs[52:104]
    hand_sizes_part = obs[104:108]
    last_play_exists_part = obs[108:109]

    print(f"✓ Hand cards: {hand_part.sum()}")
    print(f"✓ Last play cards: {last_play_part.sum()}")
    print(f"✓ Hand sizes: {hand_sizes_part}")
    print(f"✓ Last play exists: {last_play_exists_part[0]}")

    # Make a move
    action_mask = wrapper.get_action_mask()
    valid_action = np.where(action_mask)[0][0]

    obs, reward, done, truncated, info = wrapper.step(valid_action)

    # Check last play is now recorded
    last_play_part = obs[52:104]
    last_play_exists_part = obs[108:109]

    print(f"✓ After move - last play cards: {last_play_part.sum()}")
    print(f"✓ After move - last play exists: {last_play_exists_part[0]}")

    print("RL wrapper test passed!")


if __name__ == "__main__":
    test_observation_space()
    test_rl_wrapper()
    print("\n✓ All observation tests passed!")
