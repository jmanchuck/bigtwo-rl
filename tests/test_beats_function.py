#!/usr/bin/env python3
"""Comprehensive unit tests for the _beats() function in Big Two game engine."""

import pytest
import numpy as np
from bigtwo_rl.core.bigtwo import ToyBigTwoFullRules
from bigtwo_rl.core.card_utils import string_to_card


class TestBeatsFunction:
    """Test suite for the _beats() function."""

    def setup_method(self):
        """Set up fresh environment for each test."""
        self.env = ToyBigTwoFullRules(num_players=4)
        self.env.reset()

    def create_card_mask(self, card_strings):
        """Create boolean mask from list of card strings like ['3C', '4D']."""
        mask = np.zeros(52, dtype=bool)
        for card_str in card_strings:
            card_idx = string_to_card(card_str)
            mask[card_idx] = True
        return mask

    def set_last_play(self, card_strings, player=1):
        """Set the last play to given cards."""
        if card_strings:
            last_play_mask = self.create_card_mask(card_strings)
            self.env.last_play = (last_play_mask, player)
        else:
            self.env.last_play = None

    def test_single_card_same_rank_suit_comparison(self):
        """Test single cards of same rank with different suits."""
        # 3D should not beat 3C (same rank, lower suit)
        self.set_last_play(["3C"])
        candidate = self.create_card_mask(["3D"])
        assert not self.env._beats(candidate), (
            "3D should not beat 3C (same rank, lower suit)"
        )

        # 3H should beat 3C (same rank, higher suit)
        candidate = self.create_card_mask(["3H"])
        assert self.env._beats(candidate), "3H should beat 3C (same rank, higher suit)"

        # 3S should beat 3C (same rank, highest suit)
        candidate = self.create_card_mask(["3S"])
        assert self.env._beats(candidate), "3S should beat 3C (same rank, highest suit)"

    def test_single_card_rank_comparison(self):
        """Test single cards with different ranks."""
        # Higher rank should beat lower rank
        self.set_last_play(["3C"])
        candidate = self.create_card_mask(["4C"])
        assert self.env._beats(candidate), "4C should beat 3C (higher rank, same suit)"

        candidate = self.create_card_mask(["4D"])
        assert self.env._beats(candidate), "4D should beat 3C (higher rank, any suit)"

        # Lower rank should not beat higher rank
        self.set_last_play(["4C"])
        candidate = self.create_card_mask(["3C"])
        assert not self.env._beats(candidate), "3C should NOT beat 4C (lower rank)"

        candidate = self.create_card_mask(["3S"])
        assert not self.env._beats(candidate), (
            "3S should NOT beat 4C (lower rank, even highest suit)"
        )

    def test_single_card_high_cards(self):
        """Test high card comparisons (K, A, 2)."""
        # Ace should beat King
        self.set_last_play(["KC"])
        candidate = self.create_card_mask(["AC"])
        assert self.env._beats(candidate), "AC should beat KC (Ace > King)"

        # 2 should beat Ace (2 is highest in Big Two)
        self.set_last_play(["AC"])
        candidate = self.create_card_mask(["2C"])
        assert self.env._beats(candidate), "2C should beat AC (2 is highest in Big Two)"

        # 2S should beat 2C (highest possible card)
        self.set_last_play(["2C"])
        candidate = self.create_card_mask(["2S"])
        assert self.env._beats(candidate), "2S should beat 2C (highest card possible)"

        # 2D should not beat 2C (same rank, lower suit)
        candidate = self.create_card_mask(["2D"])
        assert not self.env._beats(candidate), (
            "2D should not beat 2C (same rank, lower suit)"
        )

    def test_pair_comparisons(self):
        """Test pair vs pair comparisons."""
        # Higher suit pair should beat lower suit pair
        self.set_last_play(["3C", "3D"])
        candidate = self.create_card_mask(["3H", "3S"])
        assert self.env._beats(candidate), (
            "Higher suit pair should beat lower suit pair"
        )

        # Higher rank pair should beat lower rank pair
        candidate = self.create_card_mask(["4C", "4D"])
        assert self.env._beats(candidate), (
            "Higher rank pair should beat lower rank pair"
        )

        # Lower rank pair should not beat higher rank pair
        self.set_last_play(["4C", "4D"])
        candidate = self.create_card_mask(["3H", "3S"])
        assert not self.env._beats(candidate), (
            "Lower rank pair should not beat higher rank pair"
        )

        # Ace pair should beat King pair
        self.set_last_play(["KC", "KD"])
        candidate = self.create_card_mask(["AC", "AD"])
        assert self.env._beats(candidate), "Ace pair should beat King pair"

        # 2 pair should beat Ace pair
        self.set_last_play(["AC", "AD"])
        candidate = self.create_card_mask(["2C", "2D"])
        assert self.env._beats(candidate), "2 pair should beat Ace pair"

    def test_pair_vs_single_mismatch(self):
        """Test that pairs and singles can't beat each other."""
        # Single card should not beat pair
        self.set_last_play(["3C", "3D"])
        candidate = self.create_card_mask(["4C"])
        assert not self.env._beats(candidate), "Single card should not beat pair"

        # Pair should not beat single (wrong number of cards)
        self.set_last_play(["3C"])
        candidate = self.create_card_mask(["4C", "4D"])
        assert not self.env._beats(candidate), (
            "Pair should not beat single (wrong number of cards)"
        )

    def test_trips_comparisons(self):
        """Test triple card comparisons."""
        # Higher rank trips should beat lower rank trips
        self.set_last_play(["3C", "3D", "3H"])
        candidate = self.create_card_mask(["4C", "4D", "4H"])
        assert self.env._beats(candidate), (
            "Higher rank trips should beat lower rank trips"
        )

        # Lower rank trips should not beat higher rank trips
        self.set_last_play(["4C", "4D", "4H"])
        candidate = self.create_card_mask(["3C", "3D", "3S"])
        assert not self.env._beats(candidate), (
            "Lower rank trips should not beat higher rank trips"
        )

        # Ace trips should beat King trips
        self.set_last_play(["KC", "KD", "KH"])
        candidate = self.create_card_mask(["AC", "AD", "AH"])
        assert self.env._beats(candidate), "Ace trips should beat King trips"

        # 2 trips should beat Ace trips
        self.set_last_play(["AC", "AD", "AH"])
        candidate = self.create_card_mask(["2C", "2D", "2H"])
        assert self.env._beats(candidate), "2 trips should beat Ace trips"

    def test_invalid_trips(self):
        """Test invalid trip combinations."""
        # Mixed cards should not beat trips
        self.set_last_play(["3C", "3D", "3H"])
        candidate = self.create_card_mask(["3S", "4C", "4D"])
        assert not self.env._beats(candidate), "Mixed cards should not beat trips"

    def test_pass_moves(self):
        """Test pass move validation."""
        # Pass should be allowed when there's a last play
        self.set_last_play(["4C"])
        pass_mask = np.zeros(52, dtype=bool)  # All False = pass
        assert self.env._beats(pass_mask), (
            "Pass should be allowed when there's a last play"
        )

        # Pass should not be allowed when starting new trick
        self.set_last_play([])  # No last play
        assert not self.env._beats(pass_mask), (
            "Pass should NOT be allowed when starting new trick"
        )

    def test_new_trick_starts(self):
        """Test starting new tricks (no last play)."""
        self.set_last_play([])  # No last play

        # Any single card should be allowed to start new trick
        candidate = self.create_card_mask(["3C"])
        assert self.env._beats(candidate), (
            "Any single card should be allowed to start new trick"
        )

        # Any pair should be allowed to start new trick
        candidate = self.create_card_mask(["3C", "3D"])
        assert self.env._beats(candidate), (
            "Any pair should be allowed to start new trick"
        )

        # Any trips should be allowed to start new trick
        candidate = self.create_card_mask(["KC", "KD", "KH"])
        assert self.env._beats(candidate), (
            "Any trips should be allowed to start new trick"
        )

        # Any 5-card hand should be allowed to start new trick
        candidate = self.create_card_mask(["3C", "4D", "5H", "6S", "7C"])
        assert self.env._beats(candidate), (
            "Any 5-card hand should be allowed to start new trick"
        )

    def test_invalid_hands_vs_single(self):
        """Test invalid hand combinations against singles."""
        # Set up a last play first
        self.set_last_play(["3C"])

        # Mixed pair should not beat single
        candidate = self.create_card_mask(["3C", "4D"])
        assert not self.env._beats(candidate), "Mixed pair should not beat single"

        # Random 3 cards should not beat single
        candidate = self.create_card_mask(["3C", "4D", "5H"])
        assert not self.env._beats(candidate), "Random 3 cards should not beat single"

        # Random 4 cards should not beat single
        candidate = self.create_card_mask(["3C", "4D", "5H", "6S"])
        assert not self.env._beats(candidate), "Random 4 cards should not beat single"

        # Empty hand (pass) should be allowed
        pass_mask = np.zeros(52, dtype=bool)
        assert self.env._beats(pass_mask), "Empty hand (pass) should be allowed"

    def test_five_card_straights(self):
        """Test 5-card straight comparisons."""
        # Higher straight should beat lower straight
        self.set_last_play(["3C", "4D", "5H", "6S", "7C"])
        candidate = self.create_card_mask(["4C", "5D", "6H", "7S", "8C"])
        assert self.env._beats(candidate), "Higher straight should beat lower straight"

        # Higher straight with high cards should beat lower straight
        self.set_last_play(["TC", "JD", "QH", "KS", "AC"])
        candidate = self.create_card_mask(["JC", "QD", "KH", "AS", "2C"])
        assert self.env._beats(candidate), "Higher straight should beat lower straight"

    def test_five_card_flushes(self):
        """Test 5-card flush comparisons."""
        # Higher flush should beat lower flush
        self.set_last_play(["3C", "5C", "7C", "9C", "JC"])
        candidate = self.create_card_mask(["4D", "6D", "8D", "TD", "QD"])
        assert self.env._beats(candidate), "Higher flush should beat lower flush"

    def test_five_card_type_hierarchy(self):
        """Test 5-card hand type hierarchy."""
        # Flush should beat straight
        self.set_last_play(["3C", "4D", "5H", "6S", "7C"])
        candidate = self.create_card_mask(["3H", "5H", "7H", "9H", "JH"])
        assert self.env._beats(candidate), "Flush should beat straight"

        # Straight should not beat flush (reverse test)
        self.set_last_play(["3H", "5H", "7H", "9H", "JH"])
        candidate = self.create_card_mask(["3C", "4D", "5H", "6S", "7C"])
        assert not self.env._beats(candidate), "Straight should not beat flush"

    def test_five_card_full_house(self):
        """Test full house comparisons."""
        # Higher full house should beat lower full house
        self.set_last_play(["3C", "3D", "3H", "4S", "4C"])
        candidate = self.create_card_mask(["4D", "4H", "4S", "5C", "5D"])
        assert self.env._beats(candidate), (
            "Higher full house should beat lower full house"
        )

    def test_five_card_four_of_kind(self):
        """Test four of a kind comparisons."""
        # Higher four of a kind should beat lower four of a kind
        self.set_last_play(["3C", "3D", "3H", "3S", "4C"])
        candidate = self.create_card_mask(["4D", "4H", "4S", "4C", "5D"])
        assert self.env._beats(candidate), (
            "Higher four of a kind should beat lower four of a kind"
        )

    def test_mixed_card_count_invalid(self):
        """Test that different numbers of cards can't beat each other."""
        # 3 cards should not beat 5 cards
        self.set_last_play(["3C", "3D", "3H", "3S", "4C"])
        candidate = self.create_card_mask(["5C", "5D", "5H"])
        assert not self.env._beats(candidate), "3 cards should not beat 5 cards"
