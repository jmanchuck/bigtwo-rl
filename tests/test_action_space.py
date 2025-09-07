"""Tests for action space functionality."""

import pytest
import numpy as np
from bigtwo_rl.core.action import (
    ActionMaskBuilder, BitsetFiveCardEngine, action_to_tuple,
    OFF_PASS, OFF_1, OFF_2, OFF_3, OFF_5, N_ACTIONS,
    PAIR_LUT, TRIPLE_LUT, PAIR_ID, TRIPLE_ID
)
from bigtwo_rl.core.game import Hand, HandType, LastFive


class TestActionToTuple:
    """Test action ID to tuple conversion."""

    def test_pass_action(self):
        assert action_to_tuple(OFF_PASS) == ()

    def test_single_actions(self):
        for i in range(13):
            action_id = OFF_1 + i
            assert action_to_tuple(action_id) == (i,)

    def test_pair_actions(self):
        for idx, (i, j) in enumerate(PAIR_LUT):
            if (i, j) in PAIR_ID:
                action_id = PAIR_ID[(i, j)]
                result = action_to_tuple(action_id)
                assert result == (i, j) or result == (j, i)  # Order may vary

    def test_triple_actions(self):
        for (i, j, k) in [(0, 1, 2), (3, 4, 5)]:  # Test a few
            if (i, j, k) in TRIPLE_ID:
                action_id = TRIPLE_ID[(i, j, k)]
                result = action_to_tuple(action_id)
                assert len(result) == 3
                assert set(result) == {i, j, k}

    def test_five_card_actions(self):
        # Test first five-card action
        action_id = OFF_5
        result = action_to_tuple(action_id)
        assert len(result) == 5
        assert all(0 <= slot < 13 for slot in result)

    def test_invalid_action_id(self):
        with pytest.raises((IndexError, KeyError, ValueError)):
            action_to_tuple(-1)
        with pytest.raises((IndexError, KeyError, ValueError)):
            action_to_tuple(N_ACTIONS)


class TestActionMaskBuilder:
    """Test ActionMaskBuilder functionality."""

    def setup_method(self):
        self.engine = BitsetFiveCardEngine()
        self.masker = ActionMaskBuilder(self.engine)

    def create_test_hand(self, cards, played=None):
        """Create a Hand for testing."""
        if played is None:
            played = [0] * len(cards)
        # Pad to 13 slots
        while len(cards) < 13:
            cards.append(0)
            played.append(1)
        return Hand(card=cards, played=played)

    def test_empty_hand_no_actions(self):
        """Test that empty hand generates no actions."""
        hand = self.create_test_hand([], [])
        ids = self.masker.single_and_multiples(hand)
        assert len(ids) == 0

    def test_single_card_actions(self):
        """Test single card action generation."""
        # Hand with just 3♦ (encoded as 0)
        hand = self.create_test_hand([0])
        ids = self.masker.single_and_multiples(hand)
        
        # Should have exactly one single action
        single_actions = [id for id in ids if OFF_1 <= id < OFF_2]
        assert len(single_actions) == 1
        assert single_actions[0] == OFF_1

    def test_pair_actions(self):
        """Test pair action generation."""
        # Hand with pair of 3s: 3♦ (0) and 3♣ (1)  
        # Note: rank<<2|suit encoding: 3♦=0, 3♣=1, 3♥=2, 3♠=3
        hand = self.create_test_hand([0, 1])  # Both rank=0, different suits
        ids = self.masker.single_and_multiples(hand)
        
        # Should have singles and one pair
        pair_actions = [id for id in ids if OFF_2 <= id < OFF_3]
        assert len(pair_actions) >= 1

    def test_first_play_must_include_three_diamonds(self):
        """Test that first play actions must contain 3♦."""
        # Hand with 3♦ and other cards
        hand = self.create_test_hand([0, 4, 8, 12, 16])  # 3♦, 3♣, 4♦, 4♣, 5♦
        ids = self.masker._first_play_mask(hand)
        
        # All returned actions should be valid
        assert all(0 <= id < N_ACTIONS for id in ids)
        
        # Should include at least the 3♦ single
        single_3d = OFF_1 + 0  # 3♦ is at slot 0
        assert single_3d in ids

    def test_first_play_without_three_diamonds(self):
        """Test first play when 3♦ is not available."""
        # Hand without 3♦
        hand = self.create_test_hand([4, 8, 12, 16, 20])
        ids = self.masker._first_play_mask(hand)
        
        # Should return no actions since 3♦ is required
        assert len(ids) == 0

    def test_first_play_three_diamonds_already_played(self):
        """Test first play when 3♦ is already played."""
        hand = self.create_test_hand([0, 4], [1, 0])  # 3♦ played, 3♣ available
        ids = self.masker._first_play_mask(hand)
        
        # Should return no actions since 3♦ is required but played
        assert len(ids) == 0

    def test_full_mask_indices_first_play(self):
        """Test full mask generation for first play."""
        hand = self.create_test_hand([0, 4, 8])  # Has 3♦
        ids = self.masker.full_mask_indices(
            hand=hand,
            last_played_cards=None,
            pass_allowed=False,
            is_first_play=True,
            has_control=False
        )
        
        # Should not include pass action (not allowed)
        assert OFF_PASS not in ids
        
        # Should have some actions (at least 3♦ single)
        assert len(ids) > 0
        assert all(0 <= id < N_ACTIONS for id in ids)

    def test_full_mask_indices_with_pass(self):
        """Test full mask generation with pass allowed."""
        hand = self.create_test_hand([4, 8, 12])  # No 3♦
        ids = self.masker.full_mask_indices(
            hand=hand,
            last_played_cards=None,
            pass_allowed=True,
            is_first_play=False,
            has_control=True
        )
        
        # Should include pass action
        assert OFF_PASS in ids

    def test_last_play_constraints(self):
        """Test that actions respect last play constraints."""
        hand = self.create_test_hand([0, 4, 8, 12, 16])  # Mix of cards
        
        # Test with single last play - should only allow stronger singles
        last_key = (1, 0)  # 4♦ (rank=1, suit=0)
        ids = self.masker.single_and_multiples(hand, HandType.SINGLE, last_key)
        
        # Check that returned singles are stronger than (1, 0)
        single_actions = [id for id in ids if OFF_1 <= id < OFF_2]
        for action_id in single_actions:
            slot = action_id - OFF_1
            if slot < len(hand.card) and not hand.played[slot]:
                card = hand.card[slot]
                rank, suit = card >> 2, card & 3
                assert (rank, suit) > last_key


class TestObservationIntegration:
    """Test observation space integration."""

    def test_basic_observation_builder_import(self):
        """Test that observation builder can be imported."""
        from bigtwo_rl.core.observation import BasicObservationBuilder
        builder = BasicObservationBuilder()
        assert builder is not None
        assert builder.observation_size == 168


class TestEngines:
    """Test five-card engines."""

    def test_bitset_engine_creation(self):
        """Test BitsetFiveCardEngine creation."""
        engine = BitsetFiveCardEngine()
        assert engine is not None

    def test_engine_generate_empty_hand(self):
        """Test engine with empty hand."""
        engine = BitsetFiveCardEngine()
        hand = Hand(card=[0] * 13, played=[1] * 13)  # All cards played
        
        # Should generate no combinations
        combinations = list(engine.generate(hand, None))
        assert len(combinations) == 0

    def test_engine_generate_full_hand(self):
        """Test engine with full hand."""
        engine = BitsetFiveCardEngine()
        
        # Create hand with 13 different cards
        cards = list(range(13))  # Different card codes
        hand = Hand(card=cards, played=[0] * 13)
        hand.build_derived()
        
        # Should generate some five-card combinations
        combinations = list(engine.generate(hand, None))
        assert len(combinations) > 0
        
        # Each combination should have 5 distinct slots
        for combo in combinations[:10]:  # Test first 10
            assert len(combo) == 5
            assert len(set(combo)) == 5  # All distinct
            assert all(0 <= slot < 13 for slot in combo)


if __name__ == "__main__":
    pytest.main([__file__])