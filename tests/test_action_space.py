"""Unit tests for Big Two fixed action space implementation."""

import pytest
import numpy as np
from itertools import combinations

from bigtwo_rl.core.action_space import (
    BigTwoActionSpace,
    BigTwoActionMasker,
    HandType,
    ActionSpec,
    create_action_space,
    create_action_masker
)


class TestBigTwoActionSpace:
    """Test the BigTwoActionSpace class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.action_space = BigTwoActionSpace()
    
    def test_total_action_count(self):
        """Test that total action count matches expected value."""
        assert len(self.action_space.actions) == BigTwoActionSpace.TOTAL_ACTIONS
        assert len(self.action_space.actions) == 1365
        
        # Verify internal consistency
        assert len(self.action_space._action_to_spec) == 1365
        assert len(self.action_space._indices_to_action) == 1365
    
    def test_action_count_breakdown(self):
        """Test that action counts by type match expected values."""
        counts = self.action_space.get_action_counts_by_type()
        
        # Expected counts based on heuristics for sorted hands
        assert counts[HandType.SINGLE] == 13  # 13 cards
        assert counts[HandType.PAIR] == 33    # Valid same-rank pair indices
        assert counts[HandType.TRIPLE] == 31  # Valid same-rank triple indices
        assert counts[HandType.FIVE_CARD] == 1287  # C(13,5) = 1287
        assert counts[HandType.PASS] == 1     # 1 pass action
        
        # Verify total
        assert sum(counts.values()) == 1365
    
    def test_singles_enumeration(self):
        """Test that all single card actions are correctly enumerated."""
        singles = self.action_space.get_actions_by_type(HandType.SINGLE)
        
        assert len(singles) == 13
        
        # Check that each single has correct indices
        for i, single in enumerate(singles):
            assert single.action_id == i
            assert single.card_indices == (i,)
            assert single.hand_type == HandType.SINGLE
            assert f"index {i}" in single.description.lower()
    
    def test_pairs_enumeration(self):
        """Test that all pair actions are correctly enumerated.""" 
        pairs = self.action_space.get_actions_by_type(HandType.PAIR)
        
        assert len(pairs) == 33
        
        # Verify expected same-rank pair patterns
        actual_pairs = [action.card_indices for action in pairs]
        
        # Check we have the expected patterns
        adjacent_pairs = [(i, i+1) for i in range(12)]  # 12 pairs
        skip1_pairs = [(i, i+2) for i in range(11)]    # 11 pairs
        skip2_pairs = [(i, i+3) for i in range(10)]    # 10 pairs
        expected_pairs = adjacent_pairs + skip1_pairs + skip2_pairs
        
        assert len(actual_pairs) == 33
        assert set(actual_pairs) == set(expected_pairs)
        
        # Check action IDs are sequential after singles
        for i, pair_action in enumerate(pairs):
            assert pair_action.action_id == 13 + i  # After 13 singles
    
    def test_triples_enumeration(self):
        """Test that all triple actions are correctly enumerated."""
        triples = self.action_space.get_actions_by_type(HandType.TRIPLE)
        
        assert len(triples) == 31
        
        # Verify expected same-rank triple patterns
        actual_triples = [action.card_indices for action in triples]
        
        # Check we have the expected patterns
        consecutive_triples = [(i, i+1, i+2) for i in range(11)]  # 11 triples
        skip_last_triples = [(i, i+1, i+3) for i in range(10)]   # 10 triples
        skip_middle_triples = [(i, i+2, i+3) for i in range(10)] # 10 triples
        expected_triples = consecutive_triples + skip_last_triples + skip_middle_triples
        
        assert len(actual_triples) == 31
        assert set(actual_triples) == set(expected_triples)
        assert all(len(triple) == 3 for triple in actual_triples)
        
        # Check action IDs are sequential after singles and pairs
        for i, triple_action in enumerate(triples):
            assert triple_action.action_id == 13 + 33 + i  # After singles + pairs
    
    def test_five_card_enumeration(self):
        """Test that all five-card hand actions are correctly enumerated."""
        five_cards = self.action_space.get_actions_by_type(HandType.FIVE_CARD)
        
        assert len(five_cards) == 1287
        
        # Verify this matches C(13,5)
        from math import comb
        assert comb(13, 5) == 1287
        
        # Verify all combinations of 5 from 13 are present
        expected_five_cards = list(combinations(range(13), 5))
        actual_five_cards = [action.card_indices for action in five_cards]
        
        assert len(actual_five_cards) == len(expected_five_cards)
        assert set(actual_five_cards) == set(expected_five_cards)
        
        # Check action IDs are sequential
        for i, five_card_action in enumerate(five_cards):
            expected_id = 13 + 33 + 31 + i  # After singles + pairs + triples
            assert five_card_action.action_id == expected_id
    
    def test_pass_action(self):
        """Test that pass action is correctly enumerated."""
        pass_actions = self.action_space.get_actions_by_type(HandType.PASS)
        
        assert len(pass_actions) == 1
        
        pass_action = pass_actions[0]
        assert pass_action.action_id == 1364  # Last action (0-indexed)
        assert pass_action.card_indices == ()
        assert pass_action.hand_type == HandType.PASS
        assert "pass" in pass_action.description.lower()
    
    def test_get_action_spec(self):
        """Test getting action specifications by ID."""
        # Test first action (single 0)
        spec = self.action_space.get_action_spec(0)
        assert spec.hand_type == HandType.SINGLE
        assert spec.card_indices == (0,)
        
        # Test last action (pass)
        spec = self.action_space.get_action_spec(1364)
        assert spec.hand_type == HandType.PASS
        assert spec.card_indices == ()
        
        # Test invalid action ID
        with pytest.raises(ValueError, match="Invalid action ID"):
            self.action_space.get_action_spec(9999)
    
    def test_get_action_id(self):
        """Test getting action ID by card indices."""
        # Test single
        action_id = self.action_space.get_action_id((5,))
        assert action_id == 5
        
        # Test pair
        pair_id = self.action_space.get_action_id((0, 1))
        spec = self.action_space.get_action_spec(pair_id)
        assert spec.hand_type == HandType.PAIR
        assert spec.card_indices == (0, 1)
        
        # Test pass
        pass_id = self.action_space.get_action_id(())
        assert pass_id == 1364
        
        # Test invalid combination
        with pytest.raises(ValueError, match="Invalid card indices"):
            self.action_space.get_action_id((13, 14))  # Indices > 12
    
    def test_action_ids_unique_and_sequential(self):
        """Test that action IDs are unique and sequential."""
        action_ids = [action.action_id for action in self.action_space.actions]
        
        # Should be sequential from 0 to 1364
        assert action_ids == list(range(1365))
        
        # Should be unique
        assert len(set(action_ids)) == len(action_ids)
    
    def test_factory_function(self):
        """Test the factory function."""
        action_space = create_action_space()
        assert isinstance(action_space, BigTwoActionSpace)
        assert len(action_space.actions) == 1365


class TestBigTwoActionMasker:
    """Test the BigTwoActionMasker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.action_space = BigTwoActionSpace()
        self.masker = BigTwoActionMasker(self.action_space)
    
    def test_mask_shape(self):
        """Test that action masks have correct shape."""
        player_hand = np.ones(52, dtype=bool)  # Full deck
        mask = self.masker.create_mask(player_hand)
        
        assert mask.shape == (1365,)
        assert mask.dtype == bool
    
    def test_pass_action_masking(self):
        """Test pass action masking rules."""
        player_hand = np.ones(52, dtype=bool)
        
        # Pass should be legal when not starting trick
        mask = self.masker.create_mask(
            player_hand, 
            is_starting_trick=False,
            must_play_3_diamonds=False
        )
        assert mask[1364]  # Pass action should be legal
        
        # Pass should be illegal when starting trick
        mask = self.masker.create_mask(
            player_hand,
            is_starting_trick=True
        )
        assert not mask[1364]  # Pass action should be illegal
        
        # Pass should be illegal when must play 3♦
        mask = self.masker.create_mask(
            player_hand,
            must_play_3_diamonds=True
        )
        assert not mask[1364]  # Pass action should be illegal
    
    def test_hand_availability_masking(self):
        """Test that only actions with available cards are legal."""
        # Create hand with only first 5 cards
        player_hand = np.zeros(52, dtype=bool)
        player_hand[:5] = True
        
        mask = self.masker.create_mask(player_hand, is_starting_trick=True)
        
        # Singles 0-4 should be legal
        for i in range(5):
            assert mask[i], f"Single {i} should be legal"
        
        # Singles 5-12 should be illegal (assuming simple mapping)
        for i in range(5, 13):
            assert not mask[i], f"Single {i} should be illegal"
    
    def test_3_diamonds_rule(self):
        """Test 3 of diamonds must-play rule."""
        player_hand = np.ones(52, dtype=bool)
        
        mask = self.masker.create_mask(
            player_hand,
            must_play_3_diamonds=True
        )
        
        # At least some actions should be legal (those including 3♦)
        assert mask.any(), "Some actions should be legal when must play 3♦"
        
        # Pass should be illegal
        assert not mask[1364], "Pass should be illegal when must play 3♦"
    
    def test_hand_type_matching(self):
        """Test that actions must match last played hand type."""
        player_hand = np.ones(52, dtype=bool)
        last_played = np.zeros(52, dtype=bool)
        last_played[0] = True  # Single card
        
        mask = self.masker.create_mask(
            player_hand,
            last_played_cards=last_played,
            last_played_type=HandType.SINGLE
        )
        
        # Should have some legal actions
        assert mask.any(), "Some actions should be legal"
        
        # This is a placeholder test - real implementation would verify
        # that only singles are legal when last play was a single
    
    def test_factory_function(self):
        """Test the masker factory function."""
        masker = create_action_masker(self.action_space)
        assert isinstance(masker, BigTwoActionMasker)
        assert masker.action_space is self.action_space
    
    def test_empty_hand_masking(self):
        """Test behavior with empty hand."""
        empty_hand = np.zeros(52, dtype=bool)
        
        mask = self.masker.create_mask(empty_hand)
        
        # Only pass should be legal with empty hand
        legal_actions = np.where(mask)[0]
        assert len(legal_actions) <= 1, "At most pass should be legal with empty hand"


class TestActionSpecDataClass:
    """Test the ActionSpec dataclass."""
    
    def test_action_spec_immutable(self):
        """Test that ActionSpec is immutable (frozen)."""
        spec = ActionSpec(
            action_id=0,
            hand_type=HandType.SINGLE,
            card_indices=(1,),
            description="Test"
        )
        
        # Should not be able to modify frozen dataclass
        with pytest.raises(Exception):  # FrozenInstanceError in Python 3.7+
            spec.action_id = 1
    
    def test_action_spec_equality(self):
        """Test ActionSpec equality comparison."""
        spec1 = ActionSpec(0, HandType.SINGLE, (1,), "Test")
        spec2 = ActionSpec(0, HandType.SINGLE, (1,), "Test")
        spec3 = ActionSpec(1, HandType.SINGLE, (1,), "Test")
        
        assert spec1 == spec2
        assert spec1 != spec3


class TestCombinatorics:
    """Test that our combinatorics match mathematical expectations."""
    
    def test_combination_counts(self):
        """Verify combination counts match mathematical formulas."""
        from math import comb
        
        # C(13,1) = 13 (singles)
        assert comb(13, 1) == 13
        
        # C(13,2) = 78 (all pairs, we use 31)
        assert comb(13, 2) == 78
        
        # C(13,3) = 286 (all triples, we use 33)
        assert comb(13, 3) == 286
        
        # C(13,5) = 1287 (five-card hands)
        assert comb(13, 5) == 1287
        
        # Our total: 13 + 33 + 31 + 1287 + 1 = 1365
        assert 13 + 33 + 31 + 1287 + 1 == 1365


class TestEndToEndIntegration:
    """Test end-to-end integration of action space and masker."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.action_space = create_action_space()
        self.masker = create_action_masker(self.action_space)
    
    def test_full_game_scenario(self):
        """Test a complete game scenario."""
        # Player has full hand
        player_hand = np.ones(52, dtype=bool)
        
        # Starting the game - must play 3♦
        mask = self.masker.create_mask(
            player_hand,
            must_play_3_diamonds=True,
            is_starting_trick=True
        )
        
        # Should have some legal moves, no pass
        legal_actions = np.where(mask)[0]
        assert len(legal_actions) > 0, "Should have legal moves to start game"
        assert 1364 not in legal_actions, "Pass should not be legal when must play 3♦"
    
    def test_action_space_consistency(self):
        """Test that action space and masker are consistent."""
        # Every action in action space should be maskable
        player_hand = np.ones(52, dtype=bool)
        
        mask = self.masker.create_mask(player_hand, is_starting_trick=True)
        
        # Mask should have same length as action space
        assert len(mask) == len(self.action_space.actions)
        assert len(mask) == self.action_space.TOTAL_ACTIONS
        
        # Should have some legal actions
        assert mask.any(), "Should have some legal actions with full hand"


if __name__ == "__main__":
    pytest.main([__file__])