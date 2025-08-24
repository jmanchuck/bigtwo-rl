"""Tests for MultiPlayerRolloutBuffer."""

import pytest
import numpy as np
import torch as th
from gymnasium import spaces
from bigtwo_rl.training.multi_player_buffer_enhanced import MultiPlayerRolloutBuffer


class TestMultiPlayerRolloutBuffer:
    """Test suite for MultiPlayerRolloutBuffer."""
    
    @pytest.fixture
    def buffer_setup(self):
        """Setup buffer for testing."""
        obs_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        action_space = spaces.Discrete(5)
        
        buffer = MultiPlayerRolloutBuffer(
            buffer_size=100,
            observation_space=obs_space,
            action_space=action_space,
            device="cpu",
            gae_lambda=0.95,
            gamma=0.99,
            n_envs=2
        )
        return buffer, obs_space
    
    def test_buffer_initialization(self, buffer_setup):
        """Test buffer initializes correctly."""
        buffer, obs_space = buffer_setup
        
        assert len(buffer.transition_buffer) == 2  # n_envs
        assert all(buf.maxlen == 4 for buf in buffer.transition_buffer)
        assert len(buffer.pending_rewards) == 2
        assert buffer.delayed_rewards_assigned == 0
        assert buffer.games_completed == 0
    
    def test_normal_step_addition(self, buffer_setup):
        """Test adding normal steps (not game-ending)."""
        buffer, obs_space = buffer_setup
        
        # Add a normal step
        obs = np.random.random((2, 10)).astype(np.float32)
        action = np.array([1, 2])
        reward = np.array([0.0, 0.0])  # Normal zero rewards
        episode_start = np.array([False, False])
        value = th.tensor([0.5, 0.6])
        log_prob = th.tensor([-0.1, -0.2])
        
        initial_size = buffer.pos
        buffer.add(obs, action, reward, episode_start, value, log_prob)
        
        # Should add to transition buffer and main buffer
        assert len(buffer.transition_buffer[0]) == 1
        assert len(buffer.transition_buffer[1]) == 1
        assert buffer.pos == initial_size + 1
    
    def test_delayed_reward_assignment(self, buffer_setup):
        """Test delayed reward assignment when game ends."""
        buffer, obs_space = buffer_setup
        
        # Add 3 normal steps to build up transition buffer
        for i in range(3):
            obs = np.random.random((2, 10)).astype(np.float32)
            action = np.array([i, i+1])
            reward = np.array([0.0, 0.0])
            episode_start = np.array([False, False])
            value = th.tensor([0.5, 0.6])
            log_prob = th.tensor([-0.1, -0.2])
            
            buffer.add(obs, action, reward, episode_start, value, log_prob)
        
        # Now add a game-ending step with multi-player rewards
        obs = np.random.random((2, 10)).astype(np.float32)
        action = np.array([3, 4])
        game_rewards = [np.array([10.0, -2.0, -3.0, -5.0]), np.array([5.0, -1.0, -2.0, -2.0])]  # 4-player rewards
        episode_start = np.array([True, True])  # Episode ends
        value = th.tensor([0.0, 0.0])
        log_prob = th.tensor([-0.3, -0.4])
        
        initial_games = buffer.games_completed
        buffer.add(obs, action, game_rewards, episode_start, value, log_prob)
        
        # Should have processed game end rewards
        assert buffer.games_completed == initial_games + 2  # Both environments
        assert buffer.delayed_rewards_assigned > 0
        
        # Transition buffers should be cleared
        assert len(buffer.transition_buffer[0]) == 0
        assert len(buffer.transition_buffer[1]) == 0
    
    def test_multi_player_gae_computation(self, buffer_setup):
        """Test multi-player GAE computation."""
        buffer, obs_space = buffer_setup
        
        # Fill buffer with some data
        for i in range(8):  # 8 steps total
            obs = np.random.random((2, 10)).astype(np.float32)
            action = np.array([i % 5, (i+1) % 5])
            reward = np.array([1.0 if i % 4 == 0 else 0.0, 0.5 if i % 4 == 1 else 0.0])  # Rewards for different players
            episode_start = np.array([False, False])
            value = th.tensor([float(i), float(i+1)])
            log_prob = th.tensor([-0.1, -0.2])
            
            buffer.add(obs, action, reward, episode_start, value, log_prob)
        
        # Store original advantages
        original_advantages = buffer.advantages.copy()
        
        # Compute multi-player GAE
        buffer.compute_multi_player_gae(gamma=0.99, gae_lambda=0.95)
        
        # Advantages should be updated
        assert not np.array_equal(buffer.advantages, original_advantages)
        
        # Returns should be advantages + values
        expected_returns = buffer.advantages + buffer.values
        np.testing.assert_array_almost_equal(buffer.returns, expected_returns)
    
    def test_transition_buffer_maxlen(self, buffer_setup):
        """Test that transition buffer respects maxlen=4."""
        buffer, obs_space = buffer_setup
        
        # Add 6 normal steps (more than maxlen)
        for i in range(6):
            obs = np.random.random((2, 10)).astype(np.float32)
            action = np.array([i % 5, (i+1) % 5])
            reward = np.array([0.0, 0.0])
            episode_start = np.array([False, False])
            value = th.tensor([0.5, 0.6])
            log_prob = th.tensor([-0.1, -0.2])
            
            buffer.add(obs, action, reward, episode_start, value, log_prob)
        
        # Should only keep last 4 transitions
        assert len(buffer.transition_buffer[0]) == 4
        assert len(buffer.transition_buffer[1]) == 4
    
    def test_statistics(self, buffer_setup):
        """Test buffer statistics tracking."""
        buffer, obs_space = buffer_setup
        
        stats = buffer.get_statistics()
        
        assert 'delayed_rewards_assigned' in stats
        assert 'games_completed' in stats
        assert 'buffer_sizes' in stats
        assert stats['delayed_rewards_assigned'] == 0
        assert stats['games_completed'] == 0
        assert len(stats['buffer_sizes']) == 2  # n_envs
    
    def test_buffer_reset(self, buffer_setup):
        """Test buffer reset functionality."""
        buffer, obs_space = buffer_setup
        
        # Add some data
        obs = np.random.random((2, 10)).astype(np.float32)
        action = np.array([1, 2])
        reward = np.array([0.0, 0.0])
        episode_start = np.array([False, False])
        value = th.tensor([0.5, 0.6])
        log_prob = th.tensor([-0.1, -0.2])
        
        buffer.add(obs, action, reward, episode_start, value, log_prob)
        
        # Reset buffer
        buffer.reset()
        
        # Everything should be reset
        assert all(len(buf) == 0 for buf in buffer.transition_buffer)
        assert buffer.delayed_rewards_assigned == 0
        assert buffer.games_completed == 0
        assert buffer.pos == 0
    
    def test_single_environment(self):
        """Test buffer with single environment."""
        obs_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        action_space = spaces.Discrete(5)
        
        buffer = MultiPlayerRolloutBuffer(
            buffer_size=50,
            observation_space=obs_space,
            action_space=action_space,
            device="cpu",
            gae_lambda=0.95,
            gamma=0.99,
            n_envs=1
        )
        
        # Test single environment operations
        obs = np.random.random((10,)).astype(np.float32)
        action = np.array([1])  # Convert to numpy array
        reward = np.array([0.0])  # Convert to numpy array
        episode_start = np.array([False])  # Convert to numpy array
        value = th.tensor([0.5])
        log_prob = th.tensor([-0.1])
        
        buffer.add(obs, action, reward, episode_start, value, log_prob)
        
        assert len(buffer.transition_buffer[0]) == 1
        assert buffer.pos == 1
    
    def test_game_end_with_fewer_than_4_transitions(self, buffer_setup):
        """Test game ending with fewer than 4 transitions in buffer."""
        buffer, obs_space = buffer_setup
        
        # Add only 2 transitions
        for i in range(2):
            obs = np.random.random((2, 10)).astype(np.float32)
            action = np.array([i, i+1])
            reward = np.array([0.0, 0.0])
            episode_start = np.array([False, False])
            value = th.tensor([0.5, 0.6])
            log_prob = th.tensor([-0.1, -0.2])
            
            buffer.add(obs, action, reward, episode_start, value, log_prob)
        
        # Game ends
        obs = np.random.random((2, 10)).astype(np.float32)
        action = np.array([2, 3])
        game_rewards = [np.array([8.0, -2.0, -3.0, -3.0]), np.array([6.0, -2.0, -2.0, -2.0])]
        episode_start = np.array([True, True])
        value = th.tensor([0.0, 0.0])
        log_prob = th.tensor([-0.3, -0.4])
        
        buffer.add(obs, action, game_rewards, episode_start, value, log_prob)
        
        # Should handle fewer transitions gracefully
        assert buffer.games_completed == 2
        assert buffer.delayed_rewards_assigned == 4  # 2 transitions per env