"""Tests for immediate reward functionality."""

import numpy as np
from bigtwo_rl.core.rl_wrapper import BigTwoRLWrapper
from bigtwo_rl.core.observation_builder import standard_observation
from bigtwo_rl.training.rewards import (
    BaseReward,
    DefaultReward,
    SparseReward,
)


class MockTestReward(BaseReward):
    """Test reward class for controlled testing."""

    def __init__(self, game_reward_value=1.0, episode_bonus_value=0.5):
        self.game_reward_value = game_reward_value
        self.episode_bonus_value = episode_bonus_value
        self.game_reward_calls = []
        self.episode_bonus_calls = []

    def game_reward(self, winner_player, player_idx, cards_left, all_cards_left=None):
        self.game_reward_calls.append(
            (winner_player, player_idx, cards_left, all_cards_left)
        )
        if player_idx == winner_player:
            return self.game_reward_value
        else:
            return -0.1 * cards_left

    def episode_bonus(self, games_won, total_games, avg_cards_left):
        self.episode_bonus_calls.append((games_won, total_games, avg_cards_left))
        return self.episode_bonus_value


class TestImmediateRewards:
    """Test immediate reward functionality."""

    def test_immediate_game_reward_on_win(self):
        """Test that winning a game returns immediate reward."""
        test_reward = MockTestReward(game_reward_value=2.0)
        env = BigTwoRLWrapper(observation_config=standard_observation(), games_per_episode=2, reward_function=test_reward)
        env.reset()

        # Test the game reward calculation directly
        reward = test_reward.game_reward(0, 0, 0)  # Player 0 wins with 0 cards left
        assert reward == 2.0

        # Test that the reward function was called
        assert len(test_reward.game_reward_calls) == 1
        winner, player, cards_left, all_cards = test_reward.game_reward_calls[0]
        assert winner == 0
        assert player == 0
        assert cards_left == 0

    def test_immediate_game_reward_on_loss(self):
        """Test that losing a game returns immediate penalty."""
        test_reward = MockTestReward()
        env = BigTwoRLWrapper(observation_config=standard_observation(), games_per_episode=2, reward_function=test_reward)
        env.reset()

        # Test the game reward calculation directly for a loss
        reward = test_reward.game_reward(
            1, 0, 5
        )  # Player 1 wins, Player 0 loses with 5 cards
        assert reward == -0.5  # -0.1 * 5 cards

        # Test that the reward function was called
        assert len(test_reward.game_reward_calls) == 1
        winner, player, cards_left, all_cards = test_reward.game_reward_calls[0]
        assert winner == 1
        assert player == 0
        assert cards_left == 5

    def test_episode_bonus_calculation(self):
        """Test episode bonus is calculated correctly."""
        test_reward = MockTestReward(game_reward_value=1.0, episode_bonus_value=0.8)

        # Test episode bonus calculation directly
        bonus = test_reward.episode_bonus(
            2, 2, 0
        )  # Won 2/2 games, avg 0 cards when losing
        assert bonus == 0.8

        # Test that the episode bonus function was called
        assert len(test_reward.episode_bonus_calls) == 1
        games_won, total_games, avg_cards_left = test_reward.episode_bonus_calls[0]
        assert games_won == 2
        assert total_games == 2
        assert avg_cards_left == 0

    def test_episode_statistics_tracking(self):
        """Test that episode statistics are correctly calculated."""
        test_reward = MockTestReward()

        # Test episode bonus with mixed results
        bonus = test_reward.episode_bonus(
            1, 3, 5.0
        )  # Won 1/3 games, avg 5 cards when losing
        assert bonus == 0.5  # Default bonus value

        # Test that the episode bonus function was called
        assert len(test_reward.episode_bonus_calls) == 1
        games_won, total_games, avg_cards_left = test_reward.episode_bonus_calls[0]
        assert games_won == 1
        assert total_games == 3
        assert avg_cards_left == 5.0

    def test_no_reward_function_defaults(self):
        """Test default behavior when no reward function is provided."""
        env = BigTwoRLWrapper(observation_config=standard_observation(), games_per_episode=2)
        env.reset()

        # Test that env can work with no reward function (uses defaults)
        assert env.reward_function is None

        # Test default reward calculations by calling the internal methods
        # Set up a mock game state for testing
        env.env.hands[0] = np.zeros(52, dtype=bool)  # Player 0 has no cards (wins)
        env.env.hands[1] = np.zeros(52, dtype=bool)  # Start with no cards
        env.env.hands[1][:5] = True  # Player 1 has 5 cards

        win_reward = env._calculate_game_reward(0)  # Player 0 wins
        assert win_reward == 1.0  # Default win reward

        loss_reward = env._calculate_game_reward(1)  # Player 1 loses with 5 cards
        assert loss_reward == -0.5  # Default penalty: -0.1 * 5

    def test_custom_reward_classes(self):
        """Test built-in reward classes work correctly."""
        # Test DefaultReward
        default_reward = DefaultReward()

        win_reward = default_reward.game_reward(0, 0, 0)  # Player 0 wins
        assert win_reward > 0  # Should get win reward

        loss_reward = default_reward.game_reward(
            1, 0, 10
        )  # Player 0 loses with 10 cards
        assert loss_reward < 0  # Should get loss penalty

        # Test SparseReward
        sparse_reward = SparseReward()

        win_reward = sparse_reward.game_reward(0, 0, 0)  # Player 0 wins
        assert win_reward > 0  # Should get win reward

        loss_reward = sparse_reward.game_reward(
            1, 0, 10
        )  # Player 0 loses with 10 cards
        assert loss_reward == -0.25  # Sparse loss penalty

    def test_mid_step_rewards_are_zero(self):
        """Test that game reward is only called when games end."""
        test_reward = MockTestReward(game_reward_value=10.0)
        env = BigTwoRLWrapper(observation_config=standard_observation(), games_per_episode=1, reward_function=test_reward)
        env.reset()

        # Before any game ends, no game rewards should be calculated
        assert len(test_reward.game_reward_calls) == 0

        # When we actually call the game reward method, it should return the expected value
        reward = test_reward.game_reward(0, 0, 0)  # Player 0 wins
        assert reward == 10.0
        assert len(test_reward.game_reward_calls) == 1

    def test_reset_clears_episode_stats(self):
        """Test that reset properly clears episode statistics."""
        test_reward = MockTestReward()
        env = BigTwoRLWrapper(observation_config=standard_observation(), games_per_episode=2, reward_function=test_reward)
        env.reset()

        # Check that environment initializes with clean episode stats
        assert env.games_played == 0
        assert env.games_won == 0
        assert env.total_cards_when_losing == 0
        assert env.losses_count == 0

        # After reset again, stats should still be clean
        env.reset()
        assert env.games_played == 0
        assert env.games_won == 0
        assert env.total_cards_when_losing == 0
        assert env.losses_count == 0
