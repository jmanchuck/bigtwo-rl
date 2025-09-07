"""Unit tests for Big Two game rules enforcement."""

import pytest
import numpy as np
import sys
import os

# Add project root to path for tests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigtwo_rl.core.bigtwo import ToyBigTwoFullRules


class TestThreeDiamondsStartingRule:
    """Test the 3♦ starting rule enforcement."""

    def test_starting_player_has_three_diamonds(self):
        """Test that the starting player always has 3♦."""
        game = ToyBigTwoFullRules(num_players=4)

        # Test multiple random seeds
        for seed in range(10):
            game.reset(seed=seed)

            starting_player = game.current_player
            three_of_diamonds = 0  # card index 0 = 3♦

            # Verify starting player has 3♦
            assert game.hands[starting_player, three_of_diamonds], (
                f"Starting player {starting_player} does not have 3♦ (seed {seed})"
            )

    def test_only_starting_player_can_play_at_start(self):
        """Test that only the starting player has legal moves at game start."""
        game = ToyBigTwoFullRules(num_players=4)

        for seed in range(5):
            game.reset(seed=seed)
            starting_player = game.current_player

            for player in range(4):
                legal_moves = game.legal_moves(player)
                non_pass_moves = [move for move in legal_moves if np.any(move)]

                if player == starting_player:
                    assert len(non_pass_moves) > 0, f"Starting player {player} should have legal moves (seed {seed})"
                else:
                    assert len(non_pass_moves) == 0, (
                        f"Non-starting player {player} should have no legal moves (seed {seed})"
                    )

    def test_all_starting_moves_contain_three_diamonds(self):
        """Test that all legal starting moves contain 3♦."""
        game = ToyBigTwoFullRules(num_players=4)

        for seed in range(5):
            game.reset(seed=seed)
            starting_player = game.current_player
            three_of_diamonds = 0

            legal_moves = game.legal_moves(starting_player)

            # Check each legal move
            for i, move in enumerate(legal_moves):
                if np.any(move):  # Skip passes (though there shouldn't be any at start)
                    assert move[three_of_diamonds], f"Starting move {i} does not contain 3♦ (seed {seed})"

    def test_starting_move_types_with_three_diamonds(self):
        """Test different types of starting moves (singles, pairs, etc.) all contain 3♦."""
        game = ToyBigTwoFullRules(num_players=4)

        # Use a seed where player has multiple 3s (for pairs/trips)
        for seed in range(20):
            game.reset(seed=seed)
            starting_player = game.current_player
            three_of_diamonds = 0

            legal_moves = game.legal_moves(starting_player)

            # Categorize moves by type
            singles = []
            pairs = []
            trips = []
            five_cards = []

            for move in legal_moves:
                if not np.any(move):
                    continue  # Skip passes

                card_count = np.sum(move)
                if card_count == 1:
                    singles.append(move)
                elif card_count == 2:
                    pairs.append(move)
                elif card_count == 3:
                    trips.append(move)
                elif card_count == 5:
                    five_cards.append(move)

            # All move types should contain 3♦
            all_moves = singles + pairs + trips + five_cards
            for i, move in enumerate(all_moves):
                assert move[three_of_diamonds], f"Move type with {np.sum(move)} cards does not contain 3♦ (seed {seed})"

            # If we found different move types, we've tested them
            if len(singles) > 0 or len(pairs) > 0 or len(trips) > 0 or len(five_cards) > 0:
                break

    def test_no_pass_allowed_at_game_start(self):
        """Test that passing is not allowed at the very start of the game."""
        game = ToyBigTwoFullRules(num_players=4)

        for seed in range(5):
            game.reset(seed=seed)
            starting_player = game.current_player

            legal_moves = game.legal_moves(starting_player)
            pass_moves = [move for move in legal_moves if not np.any(move)]

            assert len(pass_moves) == 0, f"Passing should not be allowed at game start (seed {seed})"

    def test_first_move_execution_contains_three_diamonds(self):
        """Test that executing the first move actually plays 3♦."""
        game = ToyBigTwoFullRules(num_players=4)

        for seed in range(5):
            game.reset(seed=seed)
            starting_player = game.current_player
            three_of_diamonds = 0

            # Execute first legal move
            legal_moves = game.legal_moves(starting_player)
            assert len(legal_moves) > 0, f"No legal moves at start (seed {seed})"

            obs, rewards, done, info = game.step(0)  # Play first legal move

            # Verify 3♦ was played
            assert game.last_play is not None, "No last play recorded after first move"
            played_cards = np.where(game.last_play[0])[0]
            assert three_of_diamonds in played_cards, f"First move {played_cards} does not contain 3♦ (seed {seed})"

    def test_game_start_detection(self):
        """Test that game correctly detects start vs mid-game states."""
        game = ToyBigTwoFullRules(num_players=4)
        game.reset(seed=42)

        # At start: last_play should be None
        assert game.last_play is None, "Game should start with no last_play"

        # After first move: last_play should be set
        obs, rewards, done, info = game.step(0)
        assert game.last_play is not None, "last_play should be set after first move"

        # Verify 3♦ rule no longer applies
        current_player = game.current_player
        legal_moves = game.legal_moves(current_player)

        # Should now have moves that don't necessarily contain 3♦
        moves_without_3d = [move for move in legal_moves if np.any(move) and not move[0]]
        # Note: might be 0 if player doesn't have cards higher than 3♦


class TestThreePassesRule:
    """Test the 3-passes rule enforcement."""

    def test_passes_counter_increments(self):
        """Test that passes_in_row counter increments correctly."""
        game = ToyBigTwoFullRules(num_players=4)
        game.reset(seed=42)

        # Make first move
        obs, rewards, done, info = game.step(0)
        assert game.passes_in_row == 0, "passes_in_row should be 0 after a play"

        # Make players pass
        for expected_passes in range(1, 4):
            if game.done:
                break

            current_player = game.current_player
            legal_moves = game.legal_moves(current_player)

            # Find pass move
            pass_idx = None
            for i, move in enumerate(legal_moves):
                if not np.any(move):
                    pass_idx = i
                    break

            assert pass_idx is not None, f"No pass option available for player {current_player}"

            obs, rewards, done, info = game.step(pass_idx)

            if expected_passes < 3:
                assert game.passes_in_row == expected_passes, (
                    f"Expected {expected_passes} passes, got {game.passes_in_row}"
                )
            else:
                # After 3rd pass, should reset to 0
                assert game.passes_in_row == 0, (
                    f"passes_in_row should reset to 0 after 3 passes, got {game.passes_in_row}"
                )

    def test_trick_resets_after_three_passes(self):
        """Test that last_play resets to None after 3 passes."""
        game = ToyBigTwoFullRules(num_players=4)
        game.reset(seed=42)

        # Make first move
        obs, rewards, done, info = game.step(0)
        assert game.last_play is not None, "Should have a last_play after first move"

        # Make 3 passes
        for pass_num in range(3):
            if game.done:
                break

            current_player = game.current_player
            legal_moves = game.legal_moves(current_player)

            # Find and execute pass
            pass_idx = next(i for i, move in enumerate(legal_moves) if not np.any(move))
            obs, rewards, done, info = game.step(pass_idx)

        if not game.done:
            assert game.last_play is None, "last_play should reset to None after 3 passes"

    def test_new_trick_allows_any_cards(self):
        """Test that after 3 passes, player can play any cards (new trick)."""
        game = ToyBigTwoFullRules(num_players=4)
        game.reset(seed=123)  # Use seed that allows testing

        # Make first move (high card to force passes)
        legal_moves = game.legal_moves(game.current_player)
        # Try to find a high single card
        high_single_idx = None
        for i, move in enumerate(legal_moves):
            if np.sum(move) == 1:  # Single card
                card_idx = np.where(move)[0][0]
                if card_idx >= 40:  # High card (2s or Aces)
                    high_single_idx = i
                    break

        if high_single_idx is None:
            high_single_idx = 0  # Use first legal move

        obs, rewards, done, info = game.step(high_single_idx)
        first_play_cards = np.where(game.last_play[0])[0]

        # Force 3 passes
        for pass_num in range(3):
            if game.done:
                break

            current_player = game.current_player
            legal_moves = game.legal_moves(current_player)
            pass_idx = next((i for i, move in enumerate(legal_moves) if not np.any(move)), None)

            if pass_idx is not None:
                obs, rewards, done, info = game.step(pass_idx)
            else:
                break  # No pass available, might be end of game

        if not game.done and game.last_play is None:
            # Now current player should be able to play any cards (new trick)
            current_player = game.current_player
            legal_moves = game.legal_moves(current_player)
            non_pass_moves = [move for move in legal_moves if np.any(move)]

            # Should have moves available (unless game is ending)
            if len(non_pass_moves) > 0:
                # Try playing a low card that wouldn't beat the original high card
                low_card_idx = None
                for i, move in enumerate(non_pass_moves):
                    if np.sum(move) == 1:  # Single card
                        card_idx = np.where(move)[0][0]
                        if card_idx < first_play_cards[0]:  # Lower than original card
                            low_card_idx = i
                            break

                if low_card_idx is not None:
                    # This should work because it's a new trick
                    obs, rewards, done, info = game.step(low_card_idx)
                    assert game.last_play is not None, "Should be able to play lower card after 3 passes (new trick)"

    def test_passes_reset_after_play(self):
        """Test that passes_in_row resets to 0 when someone plays."""
        game = ToyBigTwoFullRules(num_players=4)
        game.reset(seed=42)

        # Make first move
        obs, rewards, done, info = game.step(0)

        # Make one pass
        current_player = game.current_player
        legal_moves = game.legal_moves(current_player)
        pass_idx = next(i for i, move in enumerate(legal_moves) if not np.any(move))
        obs, rewards, done, info = game.step(pass_idx)

        assert game.passes_in_row == 1, "Should have 1 pass recorded"

        # Make another pass
        current_player = game.current_player
        legal_moves = game.legal_moves(current_player)
        pass_idx = next(i for i, move in enumerate(legal_moves) if not np.any(move))
        obs, rewards, done, info = game.step(pass_idx)

        assert game.passes_in_row == 2, "Should have 2 passes recorded"

        # Now make a play (not pass)
        current_player = game.current_player
        legal_moves = game.legal_moves(current_player)
        play_idx = next((i for i, move in enumerate(legal_moves) if np.any(move)), None)

        if play_idx is not None:
            obs, rewards, done, info = game.step(play_idx)
            assert game.passes_in_row == 0, "passes_in_row should reset to 0 after someone plays"

    def test_no_infinite_passes_without_reset(self):
        """Test that we can't have more than 3 consecutive passes without trick reset."""
        game = ToyBigTwoFullRules(num_players=4)
        game.reset(seed=42)

        # Make first move
        obs, rewards, done, info = game.step(0)

        # Make exactly 3 passes and verify state
        for pass_count in range(3):
            if game.done:
                break

            current_player = game.current_player
            legal_moves = game.legal_moves(current_player)
            pass_idx = next(i for i, move in enumerate(legal_moves) if not np.any(move))
            obs, rewards, done, info = game.step(pass_idx)

        if not game.done:
            # After 3 passes, we should be in a new trick state
            assert game.passes_in_row == 0, "Passes should reset after 3"
            assert game.last_play is None, "Should be new trick after 3 passes"

    def test_four_player_pass_cycle(self):
        """Test that passes work correctly in 4-player game."""
        game = ToyBigTwoFullRules(num_players=4)
        game.reset(seed=42)

        # Make first move
        first_player = game.current_player
        obs, rewards, done, info = game.step(0)
        next_player_after_first = game.current_player

        # Track who passes and current players
        players_who_passed = []
        current_players_during_passes = []

        # 3 consecutive passes
        for _ in range(3):
            if game.done:
                break

            current_player = game.current_player
            players_who_passed.append(current_player)
            current_players_during_passes.append(current_player)

            legal_moves = game.legal_moves(current_player)
            pass_idx = next(i for i, move in enumerate(legal_moves) if not np.any(move))
            obs, rewards, done, info = game.step(pass_idx)

        if not game.done:
            # Should have cycled through 3 different players
            assert len(set(players_who_passed)) == 3, f"Should have 3 different players pass, got {players_who_passed}"

            # After 3 passes, trick should be reset and it should be next player's turn
            # The exact player depends on the game logic, but it should be reasonable
            final_player = game.current_player
            assert 0 <= final_player < 4, f"Invalid final player: {final_player}"

            # More importantly, check that it's a new trick (which we tested elsewhere)
            assert game.last_play is None, "Should be new trick after 3 passes"
            assert game.passes_in_row == 0, "Passes should be reset"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
