"""Tests for MultiPlayerPPO."""

import pytest
import numpy as np
import torch as th
from gymnasium import spaces
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import gymnasium as gym

from bigtwo_rl.training.multi_player_ppo import MultiPlayerPPO
from bigtwo_rl.training.multi_player_buffer_enhanced import MultiPlayerRolloutBuffer
from bigtwo_rl.training.callbacks import MultiPlayerGAECallback


class SimpleTestEnv(gym.Env):
    """Minimal environment for testing PPO integration."""
    
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self._step_count = 0
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._step_count = 0
        obs = np.random.random(10).astype(np.float32)
        return obs, {}
        
    def step(self, action):
        self._step_count += 1
        obs = np.random.random(10).astype(np.float32)
        reward = 0.1
        done = self._step_count >= 20  # Episode length of 20
        truncated = False
        info = {}
        return obs, reward, done, truncated, info


class TestMultiPlayerPPO:
    """Test suite for MultiPlayerPPO."""
    
    @pytest.fixture
    def simple_env(self):
        """Create simple test environment."""
        return SimpleTestEnv()
    
    def test_initialization_with_enhancements(self, simple_env):
        """Test MultiPlayerPPO initializes with multi-player enhancements."""
        model = MultiPlayerPPO(
            policy="MlpPolicy",
            env=simple_env,
            n_steps=32,
            batch_size=16,
            verbose=0,
        )
        
        # Should use enhanced buffer
        assert isinstance(model.rollout_buffer, MultiPlayerRolloutBuffer)
        
        # Should have multi-player callback
        assert model.multi_player_callback is not None
        assert isinstance(model.multi_player_callback, MultiPlayerGAECallback)
        
        # Callback should reference the model
        assert model.multi_player_callback.model is model
        
    
    def test_train_method_with_enhancements(self, simple_env):
        """Test enhanced training method calls multi-player callback."""
        model = MultiPlayerPPO(
            policy="MlpPolicy",
            env=simple_env,
            n_steps=32,
            batch_size=16,
            verbose=0,
        )
        
        # Mock the callback's _on_rollout_end method
        with patch.object(model.multi_player_callback, '_on_rollout_end') as mock_rollout_end:
            # Mock the parent train method to avoid full training
            with patch.object(MultiPlayerPPO.__bases__[0], 'train') as mock_parent_train:
                model.train()
                
                # Should have called the multi-player callback
                mock_rollout_end.assert_called_once()
                
                # Should have called parent train method
                mock_parent_train.assert_called_once()
    
    def test_get_multi_player_statistics(self, simple_env):
        """Test multi-player statistics collection."""
        model = MultiPlayerPPO(
            policy="MlpPolicy",
            env=simple_env,
            n_steps=32,
            batch_size=16,
            verbose=0,
        )
        
        # Mock callback statistics
        with patch.object(model.multi_player_callback, 'get_statistics') as mock_callback_stats:
            mock_callback_stats.return_value = {'gae_recalculations': 5}
            
            # Mock buffer statistics
            with patch.object(model.rollout_buffer, 'get_statistics') as mock_buffer_stats:
                mock_buffer_stats.return_value = {
                    'games_completed': 10,
                    'delayed_rewards_assigned': 25
                }
                
                stats = model.get_multi_player_statistics()
                
                # Should include both callback and buffer stats with prefixes
                assert 'callback_gae_recalculations' in stats
                assert stats['callback_gae_recalculations'] == 5
                assert 'buffer_games_completed' in stats
                assert stats['buffer_games_completed'] == 10
                assert 'buffer_delayed_rewards_assigned' in stats
                assert stats['buffer_delayed_rewards_assigned'] == 25
    
    def test_integration_with_enhanced_buffer(self, simple_env):
        """Test integration between MultiPlayerPPO and MultiPlayerRolloutBuffer."""
        model = MultiPlayerPPO(
            policy="MlpPolicy",
            env=simple_env,
            n_steps=32,
            batch_size=16,
            verbose=0,
        )
        
        # Buffer should be properly initialized
        assert isinstance(model.rollout_buffer, MultiPlayerRolloutBuffer)
        assert hasattr(model.rollout_buffer, 'compute_multi_player_gae')
        assert hasattr(model.rollout_buffer, 'get_statistics')
        
        # Callback should reference the model
        assert model.multi_player_callback.model is model