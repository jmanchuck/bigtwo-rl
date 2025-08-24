"""Test performance improvements from optimizations."""

import time
import numpy as np
from bigtwo_rl.core.rl_wrapper import BigTwoRLWrapper
from bigtwo_rl.core.observation_builder import standard_observation


def test_caching_performance():
    """Test the performance improvement from legal moves caching."""
    print("Testing legal moves caching performance...")

    env = BigTwoRLWrapper(
        observation_config=standard_observation(), num_players=4, games_per_episode=1
    )
    env.reset()

    # Simulate typical training pattern: step() followed by get_action_mask()
    num_iterations = 100

    start_time = time.time()
    for _ in range(num_iterations):
        # This is what happens during training
        action_mask = env.get_action_mask()  # Calls legal_moves
        valid_action = np.where(action_mask)[0][0]

        obs, reward, done, truncated, info = env.step(
            valid_action
        )  # Calls legal_moves again

        if done or truncated:
            env.reset()

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"✓ Completed {num_iterations} iterations in {elapsed:.3f}s")
    print(f"✓ Average time per step: {elapsed / num_iterations * 1000:.2f}ms")

    # Test completed successfully


if __name__ == "__main__":
    test_caching_performance()
    print("\n✓ Caching performance test completed!")
