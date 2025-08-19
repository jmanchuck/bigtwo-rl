#!/usr/bin/env python3
"""Test the RL wrapper to ensure it works correctly."""

from bigtwo_rl.core.rl_wrapper import BigTwoRLWrapper
import numpy as np

def test_basic_functionality():
    """Test basic environment functionality."""
    env = BigTwoRLWrapper()
    
    print("Testing environment reset...")
    obs, info = env.reset(seed=42)
    print(f"✓ Observation shape: {obs.shape}")
    print(f"✓ Hand cards: {np.sum(obs[:52])}")
    print(f"✓ Last rank: {obs[52]}")
    print(f"✓ Hand sizes: {obs[53:57]}")
    
    print("\nTesting action mask...")
    mask = env.get_action_mask()
    legal_actions = np.sum(mask)
    print(f"✓ Legal actions: {legal_actions}")
    
    print("\nTesting step...")
    # Take first legal action
    legal_indices = np.where(mask)[0]
    action = legal_indices[0]
    obs, reward, done, truncated, info = env.step(action)
    print(f"✓ Step completed - reward: {reward}, done: {done}")
    
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_basic_functionality()