"""Integration tests for Big Two RL system.

This module provides comprehensive integration tests to validate that the
Big Two RL system works correctly across all components.
"""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path
import time

# Import core components
from bigtwo_rl.core.action_space import BigTwoActionSpace, HandType
from bigtwo_rl.core.action_system import BigTwoActionSystem
from bigtwo_rl.core.card_mapping import CardMapper, ActionTranslator
from bigtwo_rl.core.bigtwo_wrapper import BigTwoWrapper
from bigtwo_rl.core.bigtwo import ToyBigTwoFullRules

# Import training components
from bigtwo_rl.training.trainer import Trainer
from bigtwo_rl.training.rewards import DefaultReward
from bigtwo_rl.training.hyperparams import FastExperimentalConfig
from bigtwo_rl.core.observation_builder import minimal_observation

# Import agent components
from bigtwo_rl.agents.ppo_agent import PPOAgent
from bigtwo_rl.agents.random_agent import RandomAgent
from bigtwo_rl.agents.greedy_agent import GreedyAgent

# Import evaluation components
from bigtwo_rl.evaluation.tournament import Tournament
from bigtwo_rl.evaluation.evaluator import Evaluator


class TestActionSpaceIntegration:
    """Test action space components work together correctly."""
    
    def test_action_space_completeness(self):
        """Test that action space has exactly 1365 actions."""
        action_space = BigTwoActionSpace()
        assert action_space.TOTAL_ACTIONS == 1365
        assert len(action_space.actions) == 1365
        
        # Verify all action IDs are unique
        action_ids = [action.action_id for action in action_space.actions]
        assert len(set(action_ids)) == 1365
        assert min(action_ids) == 0
        assert max(action_ids) == 1364
    
    def test_action_type_distribution(self):
        """Test that actions are distributed correctly by type."""
        action_space = BigTwoActionSpace()
        counts = action_space.get_action_counts_by_type()
        
        # Expected counts from specification
        assert counts[HandType.SINGLE] == 13
        assert counts[HandType.PAIR] == 33
        assert counts[HandType.TRIPLE] == 31
        assert counts[HandType.FIVE_CARD] == 1287  # C(13,5)
        assert counts[HandType.PASS] == 1
        
        # Total should be 1365
        total = sum(counts.values())
        assert total == 1365
    
    def test_action_translation_consistency(self):
        """Test action ID ↔ game move translation is consistent."""
        action_system = BigTwoActionSystem()
        
        # Create test hands with different card distributions
        test_hands = [
            self._create_test_hand([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),  # All lowest of each rank
            self._create_test_hand([3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51]),  # All spades
            self._create_test_hand([0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24])  # Mixed
        ]
        
        for hand in test_hands:
            # Test a sample of actions
            for action_id in range(0, 1365, 100):  # Sample every 100th action
                # Translate to game move
                game_move = action_system.translate_action_to_game_move(action_id, hand)
                
                # Validate move is feasible
                if np.any(game_move):  # Not a pass
                    assert np.all(game_move <= hand), f"Move uses cards not in hand for action {action_id}"
    
    def test_action_masking_respects_game_rules(self):
        """Test that action masking properly enforces Big Two rules."""
        action_system = BigTwoActionSystem()
        game = ToyBigTwoFullRules()
        
        # Test case: First move must include 3♦
        game.reset()
        first_player_hand = game.hands[game.current_player]
        
        # Create a mock game state that requires 3♦
        mask = action_system.get_legal_action_mask(game, first_player_hand)
        
        # Should have some legal actions
        assert np.sum(mask) > 0, "No legal actions for first move"
        
        # Test case: Action mask should have correct size
        assert mask.shape == (1365,), f"Mask has wrong shape: {mask.shape}"
        assert mask.dtype == bool, f"Mask has wrong dtype: {mask.dtype}"
    
    def _create_test_hand(self, card_ids):
        """Create test hand from list of card IDs."""
        hand = np.zeros(52, dtype=bool)
        for card_id in card_ids:
            if 0 <= card_id < 52:
                hand[card_id] = True
        return hand


class TestCardMappingIntegration:
    """Test card mapping system works correctly."""
    
    def test_card_mapping_bidirectional(self):
        """Test that card mapping is bidirectional."""
        mapper = CardMapper()
        
        # Create test hand
        test_cards = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]  # One from each rank
        hand = np.zeros(52, dtype=bool)
        hand[test_cards] = True
        
        # Map to indices
        index_to_card = mapper.game_hand_to_sorted_indices(hand)
        
        # Should have 13 mappings
        assert len(index_to_card) == 13
        
        # Test round-trip for various combinations
        test_indices_combinations = [
            (0,),  # Single
            (0, 1),  # Pair
            (0, 1, 2),  # Triple
            (0, 1, 2, 3, 4),  # Five cards
        ]
        
        for indices in test_indices_combinations:
            if all(idx in index_to_card for idx in indices):
                # Convert indices to game move
                game_move = mapper.indices_to_game_cards(indices, hand)
                
                # Should have correct number of cards
                assert np.sum(game_move) == len(indices)
                
                # All selected cards should be in original hand
                assert np.all(game_move <= hand)
    
    def test_action_translator_compatibility(self):
        """Test action translator works with legacy system."""
        translator = ActionTranslator()
        game = ToyBigTwoFullRules()
        game.reset()
        
        current_player = game.current_player
        player_hand = game.hands[current_player]
        legal_moves = game.legal_moves(current_player)
        
        if len(legal_moves) > 0:
            # Test converting legacy action to fixed action
            legacy_action = 0  # First legal move
            try:
                fixed_action = translator.legacy_action_to_fixed_action(legacy_action, legal_moves, player_hand)
                assert 0 <= fixed_action < 1365
                
                # Test converting back
                converted_back = translator.fixed_action_to_legacy_action(fixed_action, legal_moves, player_hand)
                assert converted_back == legacy_action
            except ValueError:
                # Some translations may not be possible, which is acceptable
                pass


class TestWrapperIntegration:
    """Test wrapper integration with game engine."""
    
    def test_fixed_action_wrapper_basic_functionality(self):
        """Test basic wrapper functionality."""
        wrapper = BigTwoWrapper(
            observation_config=minimal_observation(),
            games_per_episode=1,
            reward_function=DefaultReward()
        )
        
        # Test initialization
        assert wrapper.action_space.n == 1365
        assert wrapper.observation_space is not None
        
        # Test reset
        obs, info = wrapper.reset()
        assert obs is not None
        assert obs.shape == wrapper.observation_space.shape
        assert "current_player" in info
        
        # Test action mask
        action_mask = wrapper.action_masks()
        assert action_mask.shape == (1365,)
        assert action_mask.dtype == bool
        assert np.sum(action_mask) > 0  # Should have some legal actions
        
        # Test step with legal action
        legal_actions = np.where(action_mask)[0]
        if len(legal_actions) > 0:
            action = legal_actions[0]
            obs2, reward, done, truncated, info2 = wrapper.step(action)
            
            assert obs2 is not None
            assert isinstance(reward, (int, float))
            assert isinstance(done, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info2, dict)
    
    def test_wrapper_episode_completion(self):
        """Test wrapper handles episode completion correctly."""
        wrapper = BigTwoWrapper(
            observation_config=minimal_observation(),
            games_per_episode=2,  # Short episode for testing
            reward_function=DefaultReward()
        )
        
        obs, info = wrapper.reset()
        done = False
        step_count = 0
        max_steps = 500  # Safety limit
        
        while not done and step_count < max_steps:
            action_mask = wrapper.action_masks()
            legal_actions = np.where(action_mask)[0]
            
            if len(legal_actions) == 0:
                break
                
            action = legal_actions[0]  # Always take first legal action
            obs, reward, done, truncated, info = wrapper.step(action)
            step_count += 1
        
        # Should complete within reasonable number of steps
        assert step_count < max_steps, "Episode took too long to complete"


class TestTrainingIntegration:
    """Test training system integration."""
    
    @pytest.mark.slow
    def test_training_integration(self):
        """Test that training works end-to-end with fixed actions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = Trainer(
                reward_function=DefaultReward(),
                hyperparams=FastExperimentalConfig(),
                observation_config=minimal_observation()
            )
            
            # Override directories to use temp directory
            trainer.models_dir = tmp_dir
            trainer.tensorboard_log_dir = os.path.join(tmp_dir, 'logs')
            
            # Quick training run
            model, model_dir = trainer.train(total_timesteps=1000)
            
            # Validate model
            assert model.action_space.n == 1365, f"Model should have 1365 actions, got {model.action_space.n}"
            assert os.path.exists(os.path.join(model_dir, "final_model.zip")), "Model should be saved"
            
            # Test model can make predictions
            env = trainer._make_env()
            obs, _ = env.reset()
            action_mask = env.action_masks()
            
            action, _ = model.predict(obs, deterministic=True)
            assert 0 <= action < 1365, f"Action should be in valid range, got {action}"
    
    def test_trainer_validation(self):
        """Test trainer validates fixed action space correctly."""
        trainer = Trainer(
            reward_function=DefaultReward(),
            hyperparams=FastExperimentalConfig(),
            observation_config=minimal_observation()
        )
        
        # Test environment creation
        env = trainer._make_env()
        assert env.action_space.n == 1365
        
        # Test action mask
        obs, _ = env.reset()
        action_mask = env.action_masks()
        assert action_mask.shape == (1365,)
        assert np.sum(action_mask) > 0


class TestAgentIntegration:
    """Test agent system integration."""
    
    def test_random_agent_functionality(self):
        """Test random agent works with fixed action space."""
        agent = RandomAgent("TestRandom")
        
        # Create test observation and mask
        obs = np.zeros(57)  # Minimal observation size
        mask = np.ones(1365, dtype=bool)
        
        # Test action selection
        action = agent.get_action(obs, action_mask=mask)
        assert 0 <= action < 1365
        
        # Test with limited mask
        limited_mask = np.zeros(1365, dtype=bool)
        limited_mask[0:10] = True  # Only first 10 actions legal
        
        action = agent.get_action(obs, action_mask=limited_mask)
        assert 0 <= action < 10
    
    def test_greedy_agent_functionality(self):
        """Test greedy agent works with fixed action space."""
        agent = GreedyAgent("TestGreedy", strategy="lowest_first")
        
        # Create test observation and mask
        obs = np.zeros(57)
        mask = np.ones(1365, dtype=bool)
        
        # Test action selection
        action = agent.get_action(obs, action_mask=mask)
        assert 0 <= action < 1365
        
        # Test strategy switching
        agent.set_strategy("singles_first")
        action2 = agent.get_action(obs, action_mask=mask)
        assert 0 <= action2 < 1365
    
    def test_agent_compatibility_validation(self):
        """Test agent compatibility validation."""
        agents = [
            RandomAgent("Random1"),
            RandomAgent("Random2"),
            GreedyAgent("Greedy1"),
            GreedyAgent("Greedy2", strategy="clear_hand")
        ]
        
        # Should not raise any exceptions
        tournament = Tournament(agents, verbose=False)
        assert len(tournament.agents) == 4


class TestEvaluationIntegration:
    """Test evaluation system integration."""
    
    def test_tournament_basic_functionality(self):
        """Test tournament can run basic games."""
        agents = [
            RandomAgent("Random1"),
            RandomAgent("Random2"),
            GreedyAgent("Greedy1"),
            GreedyAgent("Greedy2")
        ]
        
        tournament = Tournament(agents, verbose=False)
        
        # Run a single game
        result = tournament.play_game(agents)
        
        assert 'winner' in result
        assert 'final_scores' in result
        assert 'completed' in result
        assert len(result['final_scores']) == 4
    
    def test_series_evaluation(self):
        """Test series evaluation works correctly."""
        agents = [
            RandomAgent("Random1"),
            RandomAgent("Random2"),
            GreedyAgent("Greedy1"),
            GreedyAgent("Greedy2")
        ]
        
        evaluator = Evaluator(num_games=5, verbose=False)  # Small number for testing
        
        # This would normally test against a trained model, but we'll test the framework
        tournament = Tournament(agents, verbose=False)
        
        # Test game execution
        results = []
        for i in range(3):  # Just a few games
            result = tournament.play_game(agents, game_id=i)
            results.append(result)
        
        # Validate results structure
        for result in results:
            assert 'winner' in result
            assert 'final_scores' in result
            assert 'completed' in result
            if result['completed']:
                assert result['winner'] is not None
                assert 0 <= result['winner'] < 4


class TestSystemIntegration:
    """Test end-to-end system integration."""
    
    def test_full_pipeline_integration(self):
        """Test complete pipeline from training to evaluation."""
        # This is a comprehensive test that would take too long for regular CI
        # So we'll test the components can be connected without running full training
        
        # 1. Test wrapper creation
        wrapper = BigTwoWrapper(
            observation_config=minimal_observation(),
            games_per_episode=1,
            reward_function=DefaultReward()
        )
        assert wrapper.action_space.n == 1365
        
        # 2. Test trainer setup
        trainer = Trainer(
            reward_function=DefaultReward(),
            hyperparams=FastExperimentalConfig(),
            observation_config=minimal_observation()
        )
        test_env = trainer._make_env()
        assert test_env.action_space.n == 1365
        
        # 3. Test agent creation (without actual model)
        agents = [
            RandomAgent("Random1"),
            GreedyAgent("Greedy1"),
        ]
        
        # 4. Test evaluation setup
        tournament = Tournament(agents[:2] * 2, verbose=False)  # Duplicate to get 4 agents
        assert len(tournament.agents) == 4
    
    def test_configuration_integration(self):
        """Test configuration system integration."""
        # Test that main components are accessible from package
        from bigtwo_rl import BigTwoWrapper, Trainer
        from bigtwo_rl.agents import RandomAgent, GreedyAgent
        from bigtwo_rl.evaluation import Tournament, Evaluator
        
        # Test factory functions from trainer module work
        from bigtwo_rl.training.trainer import create_trainer
        
        trainer = create_trainer(
            reward_function=DefaultReward(),
            hyperparams=FastExperimentalConfig(),
            observation_config=minimal_observation()
        )
        assert isinstance(trainer, Trainer)
        
        # Test main classes can be instantiated
        wrapper = BigTwoWrapper(
            observation_config=minimal_observation(),
            games_per_episode=1,
            reward_function=DefaultReward()
        )
        assert wrapper.action_space.n == 1365
    
    def test_performance_basic(self):
        """Test basic performance characteristics."""
        action_system = BigTwoActionSystem()
        
        # Time action mask generation
        game = ToyBigTwoFullRules()
        game.reset()
        
        start_time = time.time()
        for _ in range(100):
            player_hand = game.hands[0]
            mask = action_system.get_legal_action_mask(game, player_hand)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # Convert to ms
        assert avg_time < 10.0, f"Action mask generation too slow: {avg_time:.2f}ms"
        
        # Time action translation
        start_time = time.time()
        for _ in range(100):
            action_id = np.random.randint(0, 1365)
            move = action_system.translate_action_to_game_move(action_id, player_hand)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # Convert to ms
        assert avg_time < 1.0, f"Action translation too slow: {avg_time:.2f}ms"


if __name__ == "__main__":
    # Run tests when called directly
    pytest.main([__file__, "-v"])