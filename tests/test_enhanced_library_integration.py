"""Integration tests for enhanced multi-player library components.

These tests verify that the MultiPlayerRolloutBuffer and MultiPlayerGAECallback
work together correctly as an integrated system.
"""

import pytest
import numpy as np
import torch as th
from gymnasium import spaces
from unittest.mock import Mock, patch

from bigtwo_rl.training.multi_player_buffer_enhanced import MultiPlayerRolloutBuffer
from bigtwo_rl.training.callbacks import MultiPlayerGAECallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class TestEnhancedLibraryIntegration:
    """Integration tests for the enhanced multi-player system."""
    
    @pytest.fixture
    def enhanced_system_setup(self):
        """Set up enhanced buffer and callback system."""
        obs_space = spaces.Box(low=0, high=1, shape=(20,), dtype=np.float32)
        action_space = spaces.Discrete(10)
        
        # Create enhanced buffer
        buffer = MultiPlayerRolloutBuffer(
            buffer_size=200,
            observation_space=obs_space,
            action_space=action_space,
            device="cpu",
            gae_lambda=0.95,
            gamma=0.99,
            n_envs=4
        )
        
        # Create callback
        callback = MultiPlayerGAECallback(verbose=1)
        
        # Mock model that uses the enhanced buffer
        model = Mock()
        model.rollout_buffer = buffer
        model.gamma = 0.99
        model.gae_lambda = 0.95
        
        callback.model = model
        
        return buffer, callback, model
    
    def test_end_to_end_workflow(self, enhanced_system_setup):
        """Test complete workflow: buffer collection -> GAE callback -> training ready."""
        buffer, callback, model = enhanced_system_setup
        
        # Simulate collecting experience data over multiple steps
        total_steps = 32  # 8 steps per environment × 4 environments
        
        for step in range(total_steps):
            env_idx = step % 4  # Cycle through environments
            
            # Create realistic step data
            obs = np.random.random((4, 20)).astype(np.float32)
            action = np.array([step % 10, (step+1) % 10, (step+2) % 10, (step+3) % 10])
            reward = np.array([0.1 if step % 4 == 0 else 0.0, 
                              0.1 if step % 4 == 1 else 0.0,
                              0.1 if step % 4 == 2 else 0.0,
                              0.1 if step % 4 == 3 else 0.0])
            episode_start = np.array([False, False, False, False])
            value = th.tensor([0.5 + step * 0.01, 0.6 + step * 0.01, 
                              0.7 + step * 0.01, 0.8 + step * 0.01])
            log_prob = th.tensor([-0.1, -0.2, -0.3, -0.4])
            
            # Add step to buffer
            buffer.add(obs, action, reward, episode_start, value, log_prob)
        
        # Verify buffer has data
        assert buffer.pos > 0
        assert buffer.buffer_size == 200
        
        # Store original advantages to verify they change
        original_advantages = buffer.advantages.copy()
        
        # Trigger GAE recalculation via callback
        result = callback._on_rollout_end()
        
        # Verify callback succeeded
        assert result is True
        assert callback.gae_recalculations == 1
        
        # Verify advantages were recalculated (should be different)
        assert not np.array_equal(buffer.advantages, original_advantages)
        
        # Verify returns are properly calculated (advantages + values)
        expected_returns = buffer.advantages + buffer.values
        np.testing.assert_array_almost_equal(buffer.returns, expected_returns)
        
        print("✅ End-to-end workflow test passed!")
    
    def test_game_end_reward_assignment_with_gae(self, enhanced_system_setup):
        """Test delayed reward assignment followed by multi-player GAE."""
        buffer, callback, model = enhanced_system_setup
        
        # Add some regular transitions first
        for i in range(12):  # 3 transitions per environment
            obs = np.random.random((4, 20)).astype(np.float32)
            action = np.array([i % 10, (i+1) % 10, (i+2) % 10, (i+3) % 10])
            reward = np.array([0.0, 0.0, 0.0, 0.0])  # No rewards during game
            episode_start = np.array([False, False, False, False])
            value = th.tensor([float(i * 0.1), float((i+1) * 0.1), 
                              float((i+2) * 0.1), float((i+3) * 0.1)])
            log_prob = th.tensor([-0.1, -0.2, -0.3, -0.4])
            
            buffer.add(obs, action, reward, episode_start, value, log_prob)
        
        # Now add game-ending transitions with multi-player rewards
        obs = np.random.random((4, 20)).astype(np.float32)
        action = np.array([5, 6, 7, 8])
        # Game rewards for 4-player game (winner gets positive, others negative based on cards left)
        game_rewards = [
            np.array([15.0, -3.0, -5.0, -7.0]),  # Player 1 won  
            np.array([-2.0, 13.0, -4.0, -7.0]),  # Player 2 won
            np.array([-3.0, -2.0, 12.0, -7.0]),  # Player 3 won
            np.array([-4.0, -3.0, -2.0, 9.0])    # Player 4 won
        ]
        episode_start = np.array([True, True, True, True])  # Games ended
        value = th.tensor([0.0, 0.0, 0.0, 0.0])
        log_prob = th.tensor([-0.5, -0.6, -0.7, -0.8])
        
        # This should trigger delayed reward assignment
        initial_games = buffer.games_completed
        buffer.add(obs, action, game_rewards, episode_start, value, log_prob)
        
        # Verify games were processed
        assert buffer.games_completed == initial_games + 4
        assert buffer.delayed_rewards_assigned > 0
        
        # Store advantages before GAE recalculation
        original_advantages = buffer.advantages.copy()
        
        # Now trigger multi-player GAE recalculation
        result = callback._on_rollout_end()
        
        # Verify callback worked
        assert result is True
        assert callback.gae_recalculations == 1
        
        # Advantages should be recalculated with multi-player awareness
        assert not np.array_equal(buffer.advantages, original_advantages)
        
        # Buffer statistics should show the processing
        stats = buffer.get_statistics()
        assert stats['delayed_rewards_assigned'] > 0
        assert stats['games_completed'] > 0
        
        print("✅ Game-end reward assignment + GAE test passed!")
    
    def test_callback_integration_with_multiple_rollouts(self, enhanced_system_setup):
        """Test callback works correctly across multiple training rollouts."""
        buffer, callback, model = enhanced_system_setup
        
        # Simulate multiple training rollouts
        for rollout_num in range(3):
            # Reset buffer for new rollout
            buffer.reset()
            
            # Fill buffer with experience
            for step in range(20):  # 5 steps per environment
                obs = np.random.random((4, 20)).astype(np.float32)
                action = np.array([step % 10, (step+1) % 10, (step+2) % 10, (step+3) % 10])
                # Vary rewards per rollout
                reward = np.array([0.1 * rollout_num if step % 4 == 0 else 0.0,
                                  0.1 * rollout_num if step % 4 == 1 else 0.0,
                                  0.1 * rollout_num if step % 4 == 2 else 0.0,
                                  0.1 * rollout_num if step % 4 == 3 else 0.0])
                episode_start = np.array([False, False, False, False])
                value = th.tensor([float(step + rollout_num), float(step + rollout_num + 1),
                                  float(step + rollout_num + 2), float(step + rollout_num + 3)])
                log_prob = th.tensor([-0.1, -0.2, -0.3, -0.4])
                
                buffer.add(obs, action, reward, episode_start, value, log_prob)
            
            # Process this rollout with callback
            result = callback._on_rollout_end()
            assert result is True
        
        # Should have processed 3 rollouts
        assert callback.gae_recalculations == 3
        
        # Get callback statistics
        callback_stats = callback.get_statistics()
        assert callback_stats['gae_recalculations'] == 3
        
        print("✅ Multiple rollout integration test passed!")
    
    def test_enhanced_components_compatibility(self):
        """Test that enhanced components are compatible with stable-baselines3 patterns."""
        # This test verifies that our components follow stable-baselines3 conventions
        
        obs_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        action_space = spaces.Discrete(5)
        
        # Test buffer can be created with stable-baselines3 style parameters
        buffer = MultiPlayerRolloutBuffer(
            buffer_size=100,
            observation_space=obs_space,
            action_space=action_space,
            device="cpu"
        )
        
        # Test callback can be created and used
        callback = MultiPlayerGAECallback(verbose=0)
        
        # Test buffer interface matches expected methods
        required_methods = ['add', 'reset', 'get', 'compute_returns_and_advantage']
        for method in required_methods:
            if method == 'get' or method == 'compute_returns_and_advantage':
                # These are inherited from parent class
                assert hasattr(buffer, method)
            else:
                # These are our implementations
                assert hasattr(buffer, method)
        
        # Test buffer has enhanced methods
        assert hasattr(buffer, 'compute_multi_player_gae')
        assert hasattr(buffer, 'get_statistics')
        
        # Test callback follows BaseCallback interface
        assert hasattr(callback, '_on_step')
        assert hasattr(callback, '_on_rollout_end')
        assert hasattr(callback, '_on_training_end')
        
        print("✅ Enhanced components compatibility test passed!")
    
    def test_system_performance_characteristics(self, enhanced_system_setup):
        """Test that the enhanced system maintains reasonable performance."""
        buffer, callback, model = enhanced_system_setup
        
        # Add a substantial amount of data to test performance
        import time
        
        start_time = time.time()
        
        # Add 100 steps of experience
        for step in range(100):
            obs = np.random.random((4, 20)).astype(np.float32)
            action = np.array([step % 10, (step+1) % 10, (step+2) % 10, (step+3) % 10])
            reward = np.array([0.01, 0.02, 0.03, 0.04])
            episode_start = np.array([False, False, False, False])
            value = th.tensor([float(step), float(step+1), float(step+2), float(step+3)])
            log_prob = th.tensor([-0.1, -0.2, -0.3, -0.4])
            
            buffer.add(obs, action, reward, episode_start, value, log_prob)
        
        add_time = time.time() - start_time
        
        # Test GAE recalculation performance
        gae_start = time.time()
        callback._on_rollout_end()
        gae_time = time.time() - gae_start
        
        # Performance should be reasonable (arbitrary but reasonable thresholds)
        assert add_time < 1.0, f"Buffer additions took too long: {add_time:.3f}s"
        assert gae_time < 0.5, f"GAE recalculation took too long: {gae_time:.3f}s"
        
        print(f"✅ Performance test passed! Add: {add_time:.3f}s, GAE: {gae_time:.3f}s")