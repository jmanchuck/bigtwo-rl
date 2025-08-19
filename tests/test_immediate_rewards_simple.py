"""Simple tests for immediate reward functionality."""

import numpy as np
import pytest
from bigtwo_rl.core.rl_wrapper import BigTwoRLWrapper
from bigtwo_rl.training.rewards import BaseReward, DefaultReward, SparseReward


class MockReward(BaseReward):
    """Mock reward class for testing."""

    def __init__(self, game_reward_val=1.0, episode_bonus_val=0.5):
        self.game_reward_val = game_reward_val
        self.episode_bonus_val = episode_bonus_val
        self.game_calls = []
        self.episode_calls = []

    def game_reward(self, winner_player, player_idx, cards_left, all_cards_left=None):
        self.game_calls.append((winner_player, player_idx, cards_left))
        if player_idx == winner_player:
            return self.game_reward_val
        else:
            return -0.1 * cards_left

    def episode_bonus(self, games_won, total_games, avg_cards_left):
        self.episode_calls.append((games_won, total_games, avg_cards_left))
        return self.episode_bonus_val


def test_immediate_reward_returned():
    """Test that immediate rewards are returned during episodes."""
    mock_reward = MockReward(game_reward_val=2.0)
    env = BigTwoRLWrapper(games_per_episode=3, reward_function=mock_reward)

    obs, info = env.reset()

    # Mock a game ending by manually calling the reward methods
    game_reward = env._calculate_game_reward(0)  # Test the method directly

    # Should call our mock reward function
    # Since we don't have a real winner, it will use default logic
    assert isinstance(game_reward, float)


def test_episode_bonus_calculation():
    """Test episode bonus calculation method."""
    mock_reward = MockReward(episode_bonus_val=1.5)
    env = BigTwoRLWrapper(games_per_episode=2, reward_function=mock_reward)

    # Manually set episode stats
    env.games_won = 1
    env.games_played = 2
    env.total_cards_when_losing = 5
    env.losses_count = 1

    bonus = env._calculate_episode_bonus()
    assert bonus == 1.5
    assert len(mock_reward.episode_calls) == 1

    games_won, total_games, avg_cards_left = mock_reward.episode_calls[0]
    assert games_won == 1
    assert total_games == 2
    assert avg_cards_left == 5.0


def test_default_reward_calculation():
    """Test default reward calculation when no custom function."""
    env = BigTwoRLWrapper(games_per_episode=2)

    # Mock winning scenario
    env.env.hands = np.zeros((4, 52), dtype=bool)  # All empty hands
    reward = env._calculate_game_reward(0)
    assert reward == 1.0  # Default win reward

    # Mock losing scenario - set first 5 positions to True
    env.env.hands[0] = np.zeros(52, dtype=bool)
    env.env.hands[0][:5] = True  # Player 0 has 5 cards
    reward = env._calculate_game_reward(0)
    assert reward == -0.5  # -0.1 * 5


def test_default_episode_bonus():
    """Test default episode bonus calculation."""
    env = BigTwoRLWrapper(games_per_episode=3)

    # High win rate scenario
    env.games_won = 2
    env.games_played = 3
    env.losses_count = 1
    env.total_cards_when_losing = 3

    bonus = env._calculate_episode_bonus()
    assert bonus == 0.5  # Win rate > 0.6


def test_reward_class_instantiation():
    """Test that reward classes can be instantiated and used."""
    default_reward = DefaultReward()

    # Test game reward
    game_reward = default_reward.game_reward(0, 0, 0)  # Winner
    assert game_reward == 1.0

    game_reward = default_reward.game_reward(1, 0, 5)  # Loser with 5 cards
    assert game_reward < 0

    # Test episode bonus - need >60% win rate
    episode_bonus = default_reward.episode_bonus(4, 5, 4.0)  # 80% win rate
    assert episode_bonus == 0.5

    episode_bonus = default_reward.episode_bonus(2, 5, 4.0)  # 40% win rate
    assert episode_bonus == 0


def test_sparse_reward_class():
    """Test SparseReward class."""
    sparse_reward = SparseReward()

    # Winner gets 1.0
    reward = sparse_reward.game_reward(0, 0, 0)
    assert reward == 1.0

    # Loser gets -0.25 regardless of cards
    reward = sparse_reward.game_reward(1, 0, 10)
    assert reward == -0.25

    # No episode bonus
    bonus = sparse_reward.episode_bonus(3, 5, 2.0)
    assert bonus == 0


def test_reset_initializes_stats():
    """Test that reset properly initializes episode statistics."""
    env = BigTwoRLWrapper(games_per_episode=2)
    env.reset()

    assert env.games_played == 0
    assert env.games_won == 0
    assert env.total_cards_when_losing == 0
    assert env.losses_count == 0


def test_environment_creation_with_reward():
    """Test environment can be created with different reward functions."""
    # String-based reward
    env1 = BigTwoRLWrapper(reward_function=DefaultReward())
    assert env1.reward_function is not None

    # No reward function
    env2 = BigTwoRLWrapper()
    assert env2.reward_function is None

    # Both should reset successfully
    env1.reset()
    env2.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
