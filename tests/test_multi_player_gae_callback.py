"""Tests for MultiPlayerGAECallback."""

import pytest
import numpy as np
import torch as th
from gymnasium import spaces
from unittest.mock import Mock, MagicMock
from bigtwo_rl.training.callbacks import MultiPlayerGAECallback
from bigtwo_rl.training.multi_player_buffer_enhanced import MultiPlayerRolloutBuffer


class TestMultiPlayerGAECallback:
    """Test suite for MultiPlayerGAECallback."""
    
    @pytest.fixture
    def mock_model_with_enhanced_buffer(self):
        """Create a mock model with MultiPlayerRolloutBuffer."""
        obs_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        action_space = spaces.Discrete(5)
        
        # Create real buffer for testing
        buffer = MultiPlayerRolloutBuffer(
            buffer_size=100,
            observation_space=obs_space,
            action_space=action_space,
            device="cpu",
            gae_lambda=0.95,
            gamma=0.99,
            n_envs=2
        )
        
        # Mock model
        model = Mock()
        model.rollout_buffer = buffer
        model.gamma = 0.99
        model.gae_lambda = 0.95
        
        return model, buffer
    
    @pytest.fixture
    def mock_model_with_standard_buffer(self):
        """Create a mock model with standard rollout buffer (no multi-player GAE)."""
        model = Mock()
        
        # Create a mock that explicitly doesn't have compute_multi_player_gae
        class StandardBufferMock:
            def __init__(self):
                self.advantages = np.zeros(10)
                self.returns = np.zeros(10)
                self.values = np.zeros(10)
        
        standard_buffer = StandardBufferMock()
        
        model.rollout_buffer = standard_buffer
        model.gamma = 0.99
        model.gae_lambda = 0.95
        
        return model, standard_buffer
    
    def test_callback_initialization(self):
        """Test callback initializes correctly."""
        callback = MultiPlayerGAECallback(verbose=1)
        
        assert callback.gae_recalculations == 0
        assert callback.verbose == 1
    
    def test_rollout_end_with_enhanced_buffer(self, mock_model_with_enhanced_buffer):
        """Test callback triggers multi-player GAE recalculation."""
        model, buffer = mock_model_with_enhanced_buffer
        callback = MultiPlayerGAECallback(verbose=0)
        callback.model = model
        
        # Add some data to buffer first
        for i in range(8):
            obs = np.random.random((2, 10)).astype(np.float32)
            action = np.array([i % 5, (i+1) % 5])
            reward = np.array([0.1 * i, 0.1 * (i+1)])
            episode_start = np.array([False, False])
            value = th.tensor([float(i), float(i+1)])
            log_prob = th.tensor([-0.1, -0.2])
            
            buffer.add(obs, action, reward, episode_start, value, log_prob)
        
        # Store original advantages
        original_advantages = buffer.advantages.copy()
        
        # Trigger callback
        result = callback._on_rollout_end()
        
        # Should return True and increment counter
        assert result is True
        assert callback.gae_recalculations == 1
        
        # Advantages should be updated (different from original)
        assert not np.array_equal(buffer.advantages, original_advantages)
    
    def test_rollout_end_with_standard_buffer(self, mock_model_with_standard_buffer):
        """Test callback gracefully handles standard buffer."""
        model, buffer = mock_model_with_standard_buffer
        callback = MultiPlayerGAECallback(verbose=0)
        callback.model = model
        
        # Trigger callback
        result = callback._on_rollout_end()
        
        # Should return True but not increment counter
        assert result is True
        assert callback.gae_recalculations == 0
        
        # Buffer methods should not have been called
        assert not hasattr(buffer, 'compute_multi_player_gae')
    
    def test_verbose_output(self, mock_model_with_enhanced_buffer, capsys):
        """Test verbose output during GAE recalculation."""
        model, buffer = mock_model_with_enhanced_buffer
        callback = MultiPlayerGAECallback(verbose=2)
        callback.model = model
        
        # Add some data to buffer
        for i in range(4):
            obs = np.random.random((2, 10)).astype(np.float32)
            action = np.array([i, i+1])
            reward = np.array([0.5, 0.6])
            episode_start = np.array([False, False])
            value = th.tensor([1.0, 1.1])
            log_prob = th.tensor([-0.1, -0.2])
            
            buffer.add(obs, action, reward, episode_start, value, log_prob)
        
        # Trigger callback
        callback._on_rollout_end()
        
        # Check verbose output
        captured = capsys.readouterr()
        assert "Recalculating GAE for multi-player" in captured.out
        assert "Advantages:" in captured.out
        assert "Returns:" in captured.out
    
    def test_training_end_statistics(self, mock_model_with_enhanced_buffer, capsys):
        """Test training end statistics."""
        model, buffer = mock_model_with_enhanced_buffer
        callback = MultiPlayerGAECallback(verbose=1)
        callback.model = model
        
        # Simulate multiple rollout ends
        for _ in range(3):
            # Add some data
            obs = np.random.random((2, 10)).astype(np.float32)
            action = np.array([1, 2])
            reward = np.array([0.0, 0.0])
            episode_start = np.array([False, False])
            value = th.tensor([0.5, 0.6])
            log_prob = th.tensor([-0.1, -0.2])
            
            buffer.add(obs, action, reward, episode_start, value, log_prob)
            callback._on_rollout_end()
        
        # Trigger training end
        callback._on_training_end()
        
        # Check statistics
        captured = capsys.readouterr()
        assert "GAE recalculations performed: 3" in captured.out
        assert callback.gae_recalculations == 3
    
    def test_get_statistics(self, mock_model_with_enhanced_buffer):
        """Test statistics retrieval."""
        model, buffer = mock_model_with_enhanced_buffer
        callback = MultiPlayerGAECallback(verbose=0)
        callback.model = model
        
        # Initially no recalculations
        stats = callback.get_statistics()
        assert stats['gae_recalculations'] == 0
        
        # Add data and trigger callback
        obs = np.random.random((2, 10)).astype(np.float32)
        action = np.array([1, 2])
        reward = np.array([0.0, 0.0])
        episode_start = np.array([False, False])
        value = th.tensor([0.5, 0.6])
        log_prob = th.tensor([-0.1, -0.2])
        
        buffer.add(obs, action, reward, episode_start, value, log_prob)
        callback._on_rollout_end()
        
        # Should show 1 recalculation
        stats = callback.get_statistics()
        assert stats['gae_recalculations'] == 1
    
    def test_integration_with_buffer_gae_method(self, mock_model_with_enhanced_buffer):
        """Test integration with buffer's compute_multi_player_gae method."""
        model, buffer = mock_model_with_enhanced_buffer
        callback = MultiPlayerGAECallback(verbose=0)
        callback.model = model
        
        # Fill buffer with meaningful data that will show GAE differences
        for i in range(12):  # Enough for all 4 players to have multiple steps
            obs = np.random.random((2, 10)).astype(np.float32)
            action = np.array([i % 5, (i+1) % 5])
            # Create different rewards for different player positions
            reward = np.array([1.0 if i % 4 == 0 else 0.1, 0.8 if i % 4 == 1 else 0.1])
            episode_start = np.array([False, False])
            value = th.tensor([float(i * 0.1), float((i+1) * 0.1)])
            log_prob = th.tensor([-0.1, -0.2])
            
            buffer.add(obs, action, reward, episode_start, value, log_prob)
        
        # Store original advantages and returns
        original_advantages = buffer.advantages.copy()
        original_returns = buffer.returns.copy()
        
        # Trigger callback
        callback._on_rollout_end()
        
        # GAE should have been recalculated
        assert callback.gae_recalculations == 1
        
        # Advantages should be different
        assert not np.array_equal(buffer.advantages, original_advantages)
        
        # Returns should be advantages + values
        expected_returns = buffer.advantages + buffer.values
        np.testing.assert_array_almost_equal(buffer.returns, expected_returns)
    
    def test_multiple_rollout_ends(self, mock_model_with_enhanced_buffer):
        """Test callback handles multiple rollout ends correctly."""
        model, buffer = mock_model_with_enhanced_buffer
        callback = MultiPlayerGAECallback(verbose=0)
        callback.model = model
        
        # Simulate multiple training iterations
        for iteration in range(5):
            # Reset buffer
            buffer.reset()
            
            # Add fresh data
            for i in range(8):
                obs = np.random.random((2, 10)).astype(np.float32)
                action = np.array([i % 5, (i+1) % 5])
                reward = np.array([0.1 * iteration, 0.1 * (iteration + 1)])
                episode_start = np.array([False, False])
                value = th.tensor([float(i), float(i+1)])
                log_prob = th.tensor([-0.1, -0.2])
                
                buffer.add(obs, action, reward, episode_start, value, log_prob)
            
            # Trigger callback
            result = callback._on_rollout_end()
            assert result is True
        
        # Should have performed 5 recalculations
        assert callback.gae_recalculations == 5