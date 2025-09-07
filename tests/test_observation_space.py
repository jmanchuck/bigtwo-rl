"""Tests for observation space functionality."""

import pytest
import numpy as np
from bigtwo_rl.core.observation import BasicObservationBuilder
from bigtwo_rl.core.game import Hand


class TestBasicObservationBuilder:
    """Test BasicObservationBuilder functionality."""

    def setup_method(self):
        self.builder = BasicObservationBuilder()

    def test_observation_size(self):
        """Test observation size calculation."""
        assert self.builder.observation_size == 168

    def test_hand_encoding_empty_hand(self):
        """Test hand encoding with empty hand."""
        hand = Hand(card=[0] * 13, played=[1] * 13)  # All played
        encoded = self.builder.encode_hand(hand)
        
        assert isinstance(encoded, np.ndarray)
        assert len(encoded) == 52
        assert np.all(encoded == 0)  # All cards should be 0

    def test_hand_encoding_single_card(self):
        """Test hand encoding with single card."""
        # Hand with just 3♦ (encoded as 0)
        cards = [0] + [0] * 12
        played = [0] + [1] * 12
        hand = Hand(card=cards, played=played)
        encoded = self.builder.encode_hand(hand)
        
        assert isinstance(encoded, np.ndarray)
        assert len(encoded) == 52
        assert encoded[0] == 1  # 3♦ should be at index 0
        assert np.sum(encoded) == 1  # Only one card

    def test_hand_encoding_multiple_cards(self):
        """Test hand encoding with multiple cards."""
        # Hand with 3♦(0), 3♣(1), 3♥(2) 
        cards = [0, 1, 2] + [0] * 10
        played = [0, 0, 0] + [1] * 10
        hand = Hand(card=cards, played=played)
        encoded = self.builder.encode_hand(hand)
        
        assert len(encoded) == 52
        assert np.sum(encoded) == 3
        # Cards should be encoded at their deck positions
        assert encoded[0] == 1  # 3♦ at position 0*4+0=0
        assert encoded[1] == 1  # 3♣ at position 0*4+1=1
        assert encoded[2] == 1  # 3♥ at position 0*4+2=2

    def test_last_played_encoding_empty(self):
        """Test last played cards encoding when empty."""
        encoded = self.builder.encode_last_play([], passes=0)
        
        assert isinstance(encoded, np.ndarray)
        assert len(encoded) == 52
        assert np.all(encoded == 0)

    def test_last_played_encoding_with_passes(self):
        """Test last played cards encoding with 3+ passes."""
        encoded = self.builder.encode_last_play([0, 1], passes=3)
        
        assert len(encoded) == 52
        assert np.all(encoded == 0)  # Should be all zeros due to 3 passes

    def test_basic_functionality(self):
        """Test that the basic observation builder can be instantiated and used."""
        builder = BasicObservationBuilder()
        
        # Test with a properly sized hand (13 slots)
        cards = [0, 4, 8] + [0] * 10  # Three real cards plus padding
        played = [0, 0, 0] + [1] * 10  # Three available, rest played
        hand = Hand(card=cards, played=played)
        encoded_hand = builder.encode_hand(hand)
        
        assert isinstance(encoded_hand, np.ndarray)
        assert len(encoded_hand) == 52
        assert np.sum(encoded_hand) == 3  # Three cards


if __name__ == "__main__":
    pytest.main([__file__])