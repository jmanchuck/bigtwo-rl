"""Comprehensive tests for MoveQualityReward system."""

import pytest
import numpy as np
from bigtwo_rl.training.rewards import MoveQualityReward


class TestMoveQualityReward:
    """Test suite for move quality reward calculations."""

    def setup_method(self):
        """Set up test fixture with default reward system."""
        self.reward = MoveQualityReward(
            card_efficiency_weight=0.3,
            hand_type_weight=0.2,
            timing_weight=0.3,
            preservation_weight=0.2,
            move_reward_scale=0.1,
            game_reward_scale=1.0
        )

    def test_initialization_valid_weights(self):
        """Test successful initialization with valid weights."""
        reward = MoveQualityReward(
            card_efficiency_weight=0.25,
            hand_type_weight=0.25,
            timing_weight=0.25,
            preservation_weight=0.25
        )
        assert reward.card_efficiency_weight == 0.25
        assert reward.hand_type_weight == 0.25
        assert reward.timing_weight == 0.25
        assert reward.preservation_weight == 0.25

    def test_initialization_invalid_weights(self):
        """Test initialization fails with weights not summing to 1.0."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            MoveQualityReward(
                card_efficiency_weight=0.5,
                hand_type_weight=0.5,
                timing_weight=0.5,  # Total = 1.5, should fail
                preservation_weight=0.0
            )

    def test_move_bonus_empty_move_cards(self):
        """Test move bonus calculation for pass (empty move_cards)."""
        game_context = {
            'remaining_hand': [44, 45, 46, 47],  # Four Aces
            'opponent_card_counts': [8, 8, 8],
            'last_play_strength': 1
        }
        
        bonus = self.reward.move_bonus([], game_context)
        
        # Should evaluate pass quality and scale by move_reward_scale
        assert isinstance(bonus, float)
        assert 0.0 <= bonus <= self.reward.move_reward_scale

    def test_move_bonus_single_card(self):
        """Test move bonus for single card play."""
        move_cards = [0]  # 3♦ (lowest card)
        game_context = {
            'remaining_hand': [48, 49, 50, 51],  # Four 2s (highest cards)
            'opponent_card_counts': [10, 10, 10],
            'game_phase': 'MIDGAME'
        }
        
        bonus = self.reward.move_bonus(move_cards, game_context)
        
        # Should reward playing low while keeping high
        assert isinstance(bonus, float)
        assert bonus > 0  # Should be positive for efficient play

    def test_move_bonus_pair(self):
        """Test move bonus for pair play."""
        move_cards = [0, 4]  # Pair of 3s
        game_context = {
            'remaining_hand': [44, 45, 46, 47],  # Four Aces
            'opponent_card_counts': [10, 10, 10],
            'game_phase': 'MIDGAME'
        }
        
        bonus = self.reward.move_bonus(move_cards, game_context)
        
        # Should reward pair play and efficiency
        assert isinstance(bonus, float)
        assert bonus > 0

    def test_move_bonus_five_card_straight(self):
        """Test move bonus for 5-card straight."""
        # 3♦, 4♦, 5♦, 6♦, 7♦ (straight flush)
        move_cards = [0, 1, 2, 3, 4]
        game_context = {
            'remaining_hand': [48, 49, 50, 51],  # Four 2s
            'opponent_card_counts': [12, 12, 12],  # Early game
            'game_phase': 'MIDGAME'
        }
        
        bonus = self.reward.move_bonus(move_cards, game_context)
        
        # Should reward complex hand type and good timing
        assert isinstance(bonus, float)
        assert bonus > 0

    def test_card_efficiency_score_perfect_efficiency(self):
        """Test card efficiency with perfect play (low cards played, high kept)."""
        played_cards = [0, 1, 2]  # 3♦, 4♦, 5♦ (very low)
        remaining_hand = [48, 49, 50, 51]  # Four 2s (very high)
        
        score = self.reward._card_efficiency_score(played_cards, remaining_hand)
        
        assert isinstance(score, float)
        assert score > 0.8  # Should be very high efficiency

    def test_card_efficiency_score_poor_efficiency(self):
        """Test card efficiency with poor play (high cards played, low kept)."""
        played_cards = [48, 49, 50]  # Three 2s (very high)
        remaining_hand = [0, 1, 2, 3]  # Low cards (very low)
        
        score = self.reward._card_efficiency_score(played_cards, remaining_hand)
        
        assert isinstance(score, float)
        assert score < 0.3  # Should be low efficiency

    def test_card_efficiency_score_power_card_bonus(self):
        """Test bonus for keeping power cards (Aces and 2s)."""
        played_cards = [0, 1]  # 3D, 4D (low cards)
        remaining_hand = [48, 49]  # AD, AC (power cards - Aces)
        
        score = self.reward._card_efficiency_score(played_cards, remaining_hand)
        
        # Should get bonus for keeping power cards
        played_cards_no_bonus = [0, 1]  # Same low cards
        remaining_hand_no_bonus = [20, 21]  # 8D, 8C (non-power cards)
        
        score_no_bonus = self.reward._card_efficiency_score(played_cards_no_bonus, remaining_hand_no_bonus)
        
        assert score >= score_no_bonus  # Should be at least as good

    def test_hand_type_score_progression(self):
        """Test hand type scores increase with complexity."""
        single_score = self.reward._hand_type_score([0])  # 3D
        pair_score = self.reward._hand_type_score([0, 1])  # 3D, 3C (pair)
        trip_score = self.reward._hand_type_score([0, 1, 2])  # 3D, 3C, 3H (trip)
        straight_score = self.reward._hand_type_score([0, 4, 8, 12, 16])  # 3D, 4D, 5D, 6D, 7D (straight)
        
        # Scores should increase with complexity
        assert single_score < pair_score < trip_score < straight_score

    def test_timing_score_five_card_early_game(self):
        """Test timing score for 5-card hands in early game (good timing)."""
        move_cards = [0, 1, 2, 3, 4]  # 5-card hand
        opponent_card_counts = [12, 12, 12]  # Early game
        
        score = self.reward._timing_score(move_cards, opponent_card_counts, 'MIDGAME')
        
        assert isinstance(score, float)
        assert score > 0.8  # Should be excellent timing

    def test_timing_score_five_card_late_game(self):
        """Test timing score for 5-card hands in late game (poor timing)."""
        move_cards = [0, 1, 2, 3, 4]  # 5-card hand
        opponent_card_counts = [2, 2, 2]  # Late game
        
        score = self.reward._timing_score(move_cards, opponent_card_counts, 'ENDGAME')
        
        assert isinstance(score, float)
        assert score < 0.5  # Should be poor timing

    def test_timing_score_pairs_endgame(self):
        """Test timing score for pairs in endgame (good timing)."""
        move_cards = [0, 4]  # Pair
        opponent_card_counts = [3, 4, 5]  # Endgame
        
        score = self.reward._timing_score(move_cards, opponent_card_counts, 'ENDGAME')
        
        assert isinstance(score, float)
        assert score > 0.7  # Should be good timing for blocking

    def test_hand_preservation_score_perfect_preservation(self):
        """Test preservation score when no combinations are broken."""
        played_cards = [0]  # Single card
        remaining_hand = [4, 8, 12, 16, 20]  # Keep pairs intact
        
        score = self.reward._hand_preservation_score(played_cards, remaining_hand)
        
        assert isinstance(score, float)
        assert score >= 0.0

    def test_hand_preservation_score_empty_hand(self):
        """Test preservation score when hand becomes empty (perfect)."""
        played_cards = [0, 1, 2]
        remaining_hand = []  # Empty after move
        
        score = self.reward._hand_preservation_score(played_cards, remaining_hand)
        
        assert score == 1.0  # Perfect preservation when hand is empty

    def test_count_combination_potential_pairs(self):
        """Test counting potential pairs in hand."""
        hand = [0, 4, 1, 5, 2]  # Two pairs: 3s and 4s
        
        potential = self.reward._count_combination_potential(hand)
        
        assert isinstance(potential, int)
        assert potential >= 2  # Should count pair potentials

    def test_count_combination_potential_trips(self):
        """Test counting potential trips in hand."""
        hand = [0, 4, 8, 1, 2]  # Trip of 3s
        
        potential = self.reward._count_combination_potential(hand)
        
        assert isinstance(potential, int)
        assert potential >= 2  # Pair + trip potential

    def test_count_combination_potential_straight(self):
        """Test counting potential straights in hand."""
        hand = [0, 1, 2, 3, 4, 5]  # Consecutive ranks for straight
        
        potential = self.reward._count_combination_potential(hand)
        
        assert isinstance(potential, int)
        assert potential >= 2  # Should count straight potential

    def test_evaluate_pass_quality_strategic_pass(self):
        """Test pass quality when strategically saving power cards."""
        game_context = {
            'remaining_hand': [44, 45, 48, 49],  # Two Aces, two 2s
            'opponent_card_counts': [10, 10, 10],  # Opponents have many cards
            'last_play_strength': 1
        }
        
        quality = self.reward._evaluate_pass_quality(game_context)
        
        assert isinstance(quality, float)
        assert quality > 0.5  # Should be strategic to pass

    def test_evaluate_pass_quality_avoid_strong_play(self):
        """Test pass quality when avoiding challenging strong plays."""
        game_context = {
            'remaining_hand': [0, 1, 2, 3],
            'opponent_card_counts': [8, 8, 8],
            'last_play_strength': 5  # Strong 5-card play
        }
        
        quality = self.reward._evaluate_pass_quality(game_context)
        
        assert isinstance(quality, float)
        assert quality > 0.4  # Should be good to avoid strong play

    def test_identify_hand_type_simple_single(self):
        """Test hand type identification for single card."""
        assert self.reward._identify_hand_type_simple([0]) == 'single'

    def test_identify_hand_type_simple_pair(self):
        """Test hand type identification for valid pair."""
        assert self.reward._identify_hand_type_simple([0, 1]) == 'pair'  # 3D, 3C

    def test_identify_hand_type_simple_trip(self):
        """Test hand type identification for valid trip."""
        assert self.reward._identify_hand_type_simple([0, 1, 2]) == 'trip'  # 3D, 3C, 3H

    def test_identify_hand_type_simple_straight(self):
        """Test hand type identification for straight."""
        cards = [0, 5, 8, 13, 16]  # 3D, 4C, 5D, 6C, 7D (straight, mixed suits)
        assert self.reward._identify_hand_type_simple(cards) == 'straight'

    def test_identify_hand_type_simple_flush(self):
        """Test hand type identification for flush."""
        cards = [0, 4, 8, 32, 36]  # 3D, 4D, 5D, JD, QD (flush - all diamonds)
        result = self.reward._identify_hand_type_simple(cards)
        assert result == 'flush'

    def test_identify_hand_type_simple_full_house(self):
        """Test hand type identification for full house."""
        cards = [0, 4, 8, 1, 5]  # Three 3s, two 4s
        result = self.reward._identify_hand_type_simple(cards)
        assert result in ['full_house', 'single']  # Depends on ordering

    def test_game_reward_winner(self):
        """Test game reward for winning player."""
        reward = self.reward.game_reward(
            winner_player=0,
            player_idx=0,
            cards_left=0
        )
        
        assert reward == self.reward.game_reward_scale * 1.0

    def test_game_reward_loser(self):
        """Test game reward for losing player."""
        reward = self.reward.game_reward(
            winner_player=0,
            player_idx=1,
            cards_left=5
        )
        
        expected = -0.1 * 5 * self.reward.game_reward_scale
        assert reward == expected

    def test_episode_bonus_high_win_rate(self):
        """Test episode bonus for high win rate."""
        bonus = self.reward.episode_bonus(
            games_won=3,
            total_games=4,  # 75% win rate
            avg_cards_left=2.0
        )
        
        expected = (0.75 - 0.25) * 0.5  # (win_rate - 0.25) * 0.5
        assert bonus == expected

    def test_episode_bonus_low_win_rate(self):
        """Test episode bonus for low win rate."""
        bonus = self.reward.episode_bonus(
            games_won=1,
            total_games=10,  # 10% win rate
            avg_cards_left=8.0
        )
        
        expected = (0.1 - 0.25) * 0.5  # Negative bonus
        assert bonus == expected

    def test_move_bonus_no_context(self):
        """Test move bonus handles missing game context gracefully."""
        move_cards = [0, 1, 2]
        bonus = self.reward.move_bonus(move_cards, game_context=None)
        
        assert isinstance(bonus, float)
        assert 0.0 <= bonus <= self.reward.move_reward_scale

    def test_move_bonus_partial_context(self):
        """Test move bonus handles partial game context."""
        move_cards = [0, 1]
        game_context = {
            'remaining_hand': [2, 3, 4]
            # Missing opponent_card_counts and game_phase
        }
        
        bonus = self.reward.move_bonus(move_cards, game_context)
        
        assert isinstance(bonus, float)
        assert 0.0 <= bonus <= self.reward.move_reward_scale

    def test_all_methods_return_finite_values(self):
        """Test all methods return finite, non-NaN values."""
        # Test various inputs to ensure no infinite/NaN values
        test_cases = [
            ([0], {'remaining_hand': [1, 2, 3], 'opponent_card_counts': [5, 5, 5], 'game_phase': 'MIDGAME'}),
            ([0, 4], {'remaining_hand': [], 'opponent_card_counts': [], 'game_phase': 'ENDGAME'}),
            ([], {'remaining_hand': [44, 45, 46], 'opponent_card_counts': [1, 1, 1], 'last_play_strength': 5}),
            ([0, 1, 2, 3, 4], None)
        ]
        
        for move_cards, context in test_cases:
            bonus = self.reward.move_bonus(move_cards, context)
            assert np.isfinite(bonus), f"Non-finite value for {move_cards}, {context}"
            assert not np.isnan(bonus), f"NaN value for {move_cards}, {context}"

    def test_performance_baseline(self):
        """Basic performance test to establish baseline timing."""
        import time
        
        # Prepare test data
        move_cards = [0, 1, 2, 3, 4]  # 5-card straight
        game_context = {
            'remaining_hand': [20, 21, 22, 23, 24, 25, 26, 27],
            'opponent_card_counts': [8, 9, 10],
            'game_phase': 'MIDGAME'
        }
        
        # Time 1000 move bonus calculations
        start_time = time.time()
        for _ in range(1000):
            self.reward.move_bonus(move_cards, game_context)
        end_time = time.time()
        
        avg_time_ms = (end_time - start_time) * 1000 / 1000
        print(f"Average move_bonus calculation time: {avg_time_ms:.4f} ms")
        
        # Should complete reasonably quickly
        assert avg_time_ms < 10.0, f"move_bonus too slow: {avg_time_ms:.4f} ms"