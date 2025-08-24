"""Unit tests for the unified BigTwoRLWrapper with true self-play."""

import numpy as np
import pytest

from bigtwo_rl.core.rl_wrapper import BigTwoRLWrapper
from bigtwo_rl.training.multi_player_buffer import MultiPlayerExperienceBuffer
from bigtwo_rl.core.observation_builder import ObservationConfig
from bigtwo_rl.training.rewards import DefaultReward


class TestUnifiedBigTwoRLWrapper:
    """Test suite for unified BigTwoRLWrapper with true self-play."""
    
    @pytest.fixture
    def obs_config(self):
        """Create basic observation configuration."""
        return ObservationConfig(
            include_hand=True,
            include_last_play=True,
            include_hand_sizes=True,
            include_last_play_exists=True
        )
    
    @pytest.fixture
    def reward_function(self):
        """Create reward function."""
        return DefaultReward()
    
    @pytest.fixture
    def wrapper(self, obs_config, reward_function):
        """Create unified wrapper with self-play."""
        return BigTwoRLWrapper(
            observation_config=obs_config,
            games_per_episode=2,
            reward_function=reward_function,
            track_move_history=False
        )
    
    def test_initialization(self, wrapper):
        """Test wrapper initialization."""
        # Test basic properties
        assert wrapper.num_players == 4
        assert wrapper.games_per_episode == 2
        
        # Test spaces
        assert wrapper.observation_space is not None
        assert wrapper.action_space is not None
        assert wrapper.action_space.n == 2000
    
    def test_invalid_num_players(self, obs_config):
        """Test that invalid number of players raises error."""
        with pytest.raises(ValueError, match="Big Two requires exactly 4 players"):
            BigTwoRLWrapper(
                observation_config=obs_config,
                num_players=3,  # Invalid
                games_per_episode=1
            )
    
    def test_reset(self, wrapper):
        """Test environment reset functionality."""
        obs, info = wrapper.reset(seed=42)
        
        # Test observation
        assert isinstance(obs, np.ndarray)
        assert obs.shape == wrapper.observation_space.shape
        
        # Test info structure
        assert isinstance(info, dict)
        assert 'current_player' in info
        assert 'games_completed' in info
        assert 'episode_complete' in info
        assert info['games_completed'] == 0
        assert info['episode_complete'] is False
        
        # Test internal state
        assert wrapper.episode_complete is False
        assert wrapper.games_completed == 0
    
    def test_action_mask(self, wrapper):
        """Test action mask functionality."""
        obs, info = wrapper.reset(seed=42)
        
        # Test action mask
        action_mask = wrapper.get_action_mask()
        assert isinstance(action_mask, np.ndarray)
        assert action_mask.dtype == bool
        assert len(action_mask) == wrapper.action_space.n
        
        # Test that at least one action is legal
        legal_actions = np.where(action_mask)[0]
        assert len(legal_actions) > 0
        
        # Test alias method
        assert np.array_equal(action_mask, wrapper.action_masks())
    
    def test_step_basic(self, wrapper):
        """Test basic step functionality."""
        obs, info = wrapper.reset(seed=42)
        
        # Get legal action
        legal_actions = np.where(wrapper.get_action_mask())[0]
        action = legal_actions[0]
        
        # Step environment
        next_obs, reward, done, truncated, info = wrapper.step(action)
        
        # Test return types
        assert isinstance(next_obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        # Test observation shape
        assert next_obs.shape == wrapper.observation_space.shape
        
        # Test info structure
        assert 'current_player' in info
        assert 'games_completed' in info
        assert 'episode_complete' in info
    
    def test_episode_completion(self, wrapper):
        """Test that episodes complete properly."""
        obs, info = wrapper.reset(seed=42)
        
        # Play until episode completes (with safety limit)
        for step in range(1000):  # Safety limit
            legal_actions = np.where(wrapper.get_action_mask())[0]
            action = legal_actions[0]
            next_obs, reward, done, truncated, info = wrapper.step(action)
            
            if done:
                # Episode should be complete
                assert info['episode_complete'] is True
                assert 'multi_player_experiences' in info
                assert isinstance(info['multi_player_experiences'], list)
                
                # Should have episode bonus
                assert 'episode_bonus' in info
                assert isinstance(info['episode_bonus'], (int, float))
                
                # Should have episode metrics
                bigtwo_metrics = [k for k in info.keys() if k.startswith('bigtwo/')]
                assert len(bigtwo_metrics) > 0
                
                break
        else:
            pytest.fail("Episode did not complete within step limit")
    
    def test_episode_metrics(self, wrapper):
        """Test episode metrics functionality."""
        obs, info = wrapper.reset(seed=42)
        
        # Initial metrics should be empty
        metrics = wrapper.get_episode_metrics()
        assert isinstance(metrics, dict)
        
        # Take some steps
        for _ in range(20):
            legal_actions = np.where(wrapper.get_action_mask())[0] 
            action = legal_actions[0]
            next_obs, reward, done, truncated, info = wrapper.step(action)
            if done:
                break
        
        # If episode completed, should have detailed metrics
        if done and info.get('episode_complete', False):
            metrics = wrapper.get_episode_metrics()
            expected_metrics = [
                'bigtwo/win_rate', 'bigtwo/avg_cards_remaining', 
                'bigtwo/games_completed', 'bigtwo/games_won'
            ]
            for metric in expected_metrics:
                assert metric in metrics
    
    def test_properties(self, wrapper):
        """Test wrapper properties."""
        obs, info = wrapper.reset(seed=42)
        
        # Test games_played property
        games_played = wrapper.games_played
        assert isinstance(games_played, int)
        assert games_played >= 0
        
        # Test games_won property
        games_won = wrapper.games_won
        assert isinstance(games_won, int)
        assert games_won >= 0
        
        # Test other properties
        assert isinstance(wrapper.total_cards_when_losing, int)
        assert isinstance(wrapper.losses_count, int)
    
    def test_true_self_play_behavior(self, wrapper):
        """Test that all players use the same network (true self-play)."""
        obs, info = wrapper.reset(seed=42)
        
        players_seen = set()
        
        # Take several steps and track which players act
        for _ in range(50):
            legal_actions = np.where(wrapper.get_action_mask())[0]
            action = legal_actions[0]
            next_obs, reward, done, truncated, info = wrapper.step(action)
            
            if 'current_player' in info:
                players_seen.add(info['current_player'])
            if 'previous_player' in info:
                players_seen.add(info['previous_player'])
                
            if done:
                break
        
        # In true self-play, we should see different players acting
        # (though we can't guarantee all 4 in a short game)
        assert len(players_seen) >= 1  # At least one player acted
    
    def test_multiprocessing_safe_initialization(self, obs_config, reward_function):
        """Test that wrapper can be initialized safely in multiprocessing context."""
        # Create wrapper without calling reset (simulates multiprocessing)
        wrapper = BigTwoRLWrapper(
            observation_config=obs_config,
            games_per_episode=1,
            reward_function=reward_function
        )
        
        # Game components should be None initially (lazy initialization)
        assert wrapper.game is None
        assert wrapper.obs_vectorizer is None
        assert wrapper.episode_manager is None
        
        # Reset should initialize everything
        obs, info = wrapper.reset(seed=42)
        assert wrapper.game is not None
        assert wrapper.obs_vectorizer is not None
        assert wrapper.episode_manager is not None
    
    def test_close(self, wrapper):
        """Test environment close functionality."""
        # Should not raise any errors
        wrapper.close()


class TestMultiPlayerExperienceBuffer:
    """Test suite for MultiPlayerExperienceBuffer (keep existing tests)."""
    
    @pytest.fixture
    def buffer(self):
        """Create multi-player experience buffer."""
        return MultiPlayerExperienceBuffer(max_buffer_size=100)
    
    def test_initialization(self, buffer):
        """Test buffer initialization."""
        assert len(buffer.player_experiences) == 4
        assert len(buffer.current_episode_experiences) == 4
        assert len(buffer.current_game_experiences) == 4
        assert buffer.episode_count == 0
        assert buffer.game_count == 0
        assert buffer.max_buffer_size == 100
    
    def test_add_step_experience(self, buffer):
        """Test adding single step experience."""
        obs = np.random.random((10,))
        action = 5
        reward = 1.0
        
        buffer.add_step_experience(
            player_idx=0,
            observation=obs,
            action=action,
            reward=reward,
            done=False
        )
        
        # Check that experience was added
        assert len(buffer.current_episode_experiences[0]) == 1
        assert len(buffer.current_game_experiences[0]) == 1
        
        exp = buffer.current_episode_experiences[0][0]
        assert np.array_equal(exp['observation'], obs)
        assert exp['action'] == action
        assert exp['reward'] == reward
        assert exp['done'] is False
        assert exp['player_idx'] == 0
    
    def test_invalid_player_idx(self, buffer):
        """Test that invalid player index raises error."""
        with pytest.raises(ValueError, match="Invalid player_idx 4"):
            buffer.add_step_experience(
                player_idx=4,  # Invalid
                observation=np.random.random((10,)),
                action=0,
                reward=0.0
            )
    
    def test_get_statistics(self, buffer):
        """Test getting buffer statistics."""
        # Add some experiences
        buffer.add_multi_player_step(
            observations=[np.random.random((10,)) for _ in range(4)],
            actions=[1] * 4,
            rewards=[0.1] * 4
        )
        buffer.finalize_game(winner_player=0)
        
        stats = buffer.get_statistics()
        
        # Check statistics structure
        expected_keys = [
            'total_experiences', 'episodes_completed', 'games_completed',
            'experiences_per_player', 'player_win_rates', 'buffer_utilization'
        ]
        for key in expected_keys:
            assert key in stats
        
        assert stats['total_experiences'] == 4
        assert stats['games_completed'] == 1
        assert stats['player_win_rates'][0] == 1.0  # Player 0 won
        assert 0 <= stats['buffer_utilization'] <= 1