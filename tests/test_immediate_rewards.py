"""Tests for immediate reward functionality."""

import numpy as np
import pytest
from bigtwo_rl.core.rl_wrapper import BigTwoRLWrapper
from bigtwo_rl.training.rewards import BaseReward, DefaultReward, SparseReward, FunctionReward


class MockTestReward(BaseReward):
    """Test reward class for controlled testing."""
    
    def __init__(self, game_reward_value=1.0, episode_bonus_value=0.5):
        self.game_reward_value = game_reward_value
        self.episode_bonus_value = episode_bonus_value
        self.game_reward_calls = []
        self.episode_bonus_calls = []
    
    def game_reward(self, winner_player, player_idx, cards_left, all_cards_left=None):
        self.game_reward_calls.append((winner_player, player_idx, cards_left, all_cards_left))
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
        test_reward = TestReward(game_reward_value=2.0)
        env = BigTwoRLWrapper(games_per_episode=2, reward_function=test_reward)
        env.reset()
        
        # Simulate winning by setting the current player's hand to empty
        env.env.hands[0] = np.zeros(52, dtype=bool)  # Player 0 wins
        env.env.done = True
        
        # Take any action to trigger step
        obs, reward, done, truncated, info = env.step(0)
        
        # Should get immediate reward, not 0
        assert reward == 2.0
        assert not done  # Episode not done yet
        assert len(test_reward.game_reward_calls) == 1
        assert len(test_reward.episode_bonus_calls) == 0
    
    def test_immediate_game_reward_on_loss(self):
        """Test that losing a game returns immediate penalty."""
        test_reward = TestReward()
        env = BigTwoRLWrapper(games_per_episode=2, reward_function=test_reward)
        env.reset()
        
        # Simulate loss - another player wins
        env.env.hands[1] = np.zeros(52, dtype=bool)  # Player 1 wins
        env.env.hands[0] = np.ones(5, dtype=bool)  # Player 0 has 5 cards left
        env.env.done = True
        
        obs, reward, done, truncated, info = env.step(0)
        
        # Should get immediate penalty based on cards left
        assert reward == -0.5  # -0.1 * 5 cards
        assert not done  # Episode not done yet
        assert len(test_reward.game_reward_calls) == 1
    
    def test_episode_bonus_calculation(self):
        """Test episode bonus is added to final reward."""
        test_reward = TestReward(game_reward_value=1.0, episode_bonus_value=0.8)
        env = BigTwoRLWrapper(games_per_episode=2, reward_function=test_reward)
        env.reset()
        
        # Play first game - win
        env.env.hands[0] = np.zeros(52, dtype=bool)
        env.env.done = True
        obs, reward1, done1, _, _ = env.step(0)
        assert reward1 == 1.0
        assert not done1
        
        # Play second game - win (episode ends)
        env.env.hands[0] = np.zeros(52, dtype=bool)
        env.env.done = True
        obs, reward2, done2, _, _ = env.step(0)
        
        # Final reward should include episode bonus
        assert reward2 == 1.8  # 1.0 + 0.8 bonus
        assert done2  # Episode complete
        assert len(test_reward.episode_bonus_calls) == 1
        
        # Check episode bonus parameters
        games_won, total_games, avg_cards_left = test_reward.episode_bonus_calls[0]
        assert games_won == 2
        assert total_games == 2
        assert avg_cards_left == 0  # No losses, so average is 0
    
    def test_episode_statistics_tracking(self):
        """Test that episode statistics are correctly tracked."""
        test_reward = TestReward()
        env = BigTwoRLWrapper(games_per_episode=3, reward_function=test_reward)
        env.reset()
        
        # Game 1: Win
        env.env.hands[0] = np.zeros(52, dtype=bool)
        env.env.done = True
        env.step(0)
        
        # Game 2: Lose with 7 cards
        env.env.hands[1] = np.zeros(52, dtype=bool)  # Player 1 wins
        env.env.hands[0] = np.ones(7, dtype=bool)  # Player 0 has 7 cards
        env.env.done = True
        env.step(0)
        
        # Game 3: Lose with 3 cards (episode ends)
        env.env.hands[1] = np.zeros(52, dtype=bool)
        env.env.hands[0] = np.ones(3, dtype=bool)
        env.env.done = True
        obs, reward, done, _, _ = env.step(0)
        
        assert done
        
        # Check final episode bonus call
        games_won, total_games, avg_cards_left = test_reward.episode_bonus_calls[0]
        assert games_won == 1
        assert total_games == 3
        assert avg_cards_left == 5.0  # (7 + 3) / 2 losses
    
    def test_no_reward_function_defaults(self):
        """Test default behavior when no reward function is provided."""
        env = BigTwoRLWrapper(games_per_episode=2)
        env.reset()
        
        # Win first game
        env.env.hands[0] = np.zeros(52, dtype=bool)
        env.env.done = True
        obs, reward, done, _, _ = env.step(0)
        
        assert reward == 1.0  # Default win reward
        assert not done
        
        # Lose second game with 5 cards (episode ends)
        env.env.hands[1] = np.zeros(52, dtype=bool)
        env.env.hands[0] = np.ones(5, dtype=bool)
        env.env.done = True
        obs, reward, done, _, _ = env.step(0)
        
        # Should get loss penalty plus default episode bonus
        expected_reward = -0.5 + 0.5  # -0.1*5 + bonus for >60% win rate
        assert reward == expected_reward
        assert done
    
    def test_custom_reward_classes(self):
        """Test built-in reward classes work correctly."""
        # Test DefaultReward
        default_reward = DefaultReward()
        env = BigTwoRLWrapper(games_per_episode=1, reward_function=default_reward)
        env.reset()
        
        env.env.hands[0] = np.zeros(52, dtype=bool)
        env.env.done = True
        obs, reward, done, _, _ = env.step(0)
        
        assert reward > 0  # Should get win reward
        
        # Test SparseReward
        sparse_reward = SparseReward()
        env2 = BigTwoRLWrapper(games_per_episode=1, reward_function=sparse_reward)
        env2.reset()
        
        env2.env.hands[1] = np.zeros(52, dtype=bool)  # Player 1 wins
        env2.env.hands[0] = np.ones(10, dtype=bool)  # Player 0 loses
        env2.env.done = True
        obs, reward, done, _, _ = env2.step(0)
        
        assert reward == -0.25  # Sparse loss penalty
        assert done
    
    def test_function_reward_wrapper(self):
        """Test FunctionReward wrapper for legacy functions."""
        def simple_reward_func(winner, player, cards_left):
            return 5.0 if player == winner else -1.0
        
        wrapped_reward = FunctionReward(simple_reward_func)
        env = BigTwoRLWrapper(games_per_episode=1, reward_function=wrapped_reward)
        env.reset()
        
        env.env.hands[0] = np.zeros(52, dtype=bool)
        env.env.done = True
        obs, reward, done, _, _ = env.step(0)
        
        assert reward == 5.0
        assert done
    
    def test_mid_step_rewards_are_zero(self):
        """Test that mid-game steps return 0 reward."""
        test_reward = TestReward(game_reward_value=10.0)
        env = BigTwoRLWrapper(games_per_episode=1, reward_function=test_reward)
        env.reset()
        
        # Take action but don't end game
        env.env.done = False
        obs, reward, done, _, _ = env.step(0)
        
        assert reward == 0.0
        assert not done
        assert len(test_reward.game_reward_calls) == 0
    
    def test_reset_clears_episode_stats(self):
        """Test that reset properly clears episode statistics."""
        test_reward = TestReward()
        env = BigTwoRLWrapper(games_per_episode=2, reward_function=test_reward)
        env.reset()
        
        # Play one game
        env.env.hands[0] = np.zeros(52, dtype=bool)
        env.env.done = True
        env.step(0)
        
        # Reset should clear stats
        env.reset()
        assert env.games_played == 0
        assert env.games_won == 0
        assert env.total_cards_when_losing == 0
        assert env.losses_count == 0