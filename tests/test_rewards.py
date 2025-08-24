"""Test new reward structure."""

from bigtwo_rl.core.bigtwo import ToyBigTwoFullRules


def test_reward_structure():
    """Test different end-game scenarios."""
    env = ToyBigTwoFullRules(4)

    # Simulate end game scenarios
    test_cases = [
        ([0, 1, 2, 13], "Winner=0, others have 1,2,13 cards"),
        ([0, 5, 6, 10], "Winner=0, others have 5,6,10 cards"),
        ([0, 3, 4, 7], "Winner=0, others have 3,4,7 cards"),
        ([0, 11, 12, 13], "Winner=0, others have 11,12,13 cards"),
    ]

    for hand_sizes, description in test_cases:
        # Manually set hand sizes for testing with numpy arrays
        import numpy as np

        env.hands = np.zeros((4, 52), dtype=bool)
        for i, size in enumerate(hand_sizes):
            env.hands[i, :size] = True  # Fill first 'size' positions with True

        # Simulate player 0 winning
        rewards = [0] * 4
        winner = 0
        for p in range(4):
            cards_left = np.sum(env.hands[p])
            if p == winner:
                rewards[p] = 100
            else:
                if cards_left >= 10:
                    rewards[p] = -50
                elif cards_left >= 5:
                    rewards[p] = -10
                elif cards_left >= 3:
                    rewards[p] = -2
                elif cards_left >= 1:
                    rewards[p] = 0
                else:
                    rewards[p] = 100

        print(f"{description}")
        print(f"Hand sizes: {hand_sizes}")
        print(f"Rewards:    {rewards}")
        print()


if __name__ == "__main__":
    test_reward_structure()
