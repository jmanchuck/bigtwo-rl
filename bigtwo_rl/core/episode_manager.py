"""Episode management for multi-game Big Two RL training sessions."""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple


class EpisodeManager:
    """Manages multi-game episodes for Big Two RL training.

    Handles episode-level state tracking, metrics collection, and episode termination logic.
    Separates episode management concerns from the core RL wrapper.
    """

    def __init__(
        self,
        games_per_episode: int = 10,
        num_players: int = 4,
        controlled_player: int = 0,
    ):
        """Initialize episode manager.

        Args:
            games_per_episode: Number of games per training episode
            num_players: Number of players in each game
            controlled_player: Index of the player being trained
        """
        self.games_per_episode = games_per_episode
        self.num_players = num_players
        self.controlled_player = controlled_player

        # Episode-level tracking state
        self.games_played = 0
        self.games_won = 0
        self.total_cards_when_losing = 0
        self.losses_count = 0
        self._accumulated_move_bonuses = 0.0

        # Big Two metrics tracking
        self._episode_steps = 0
        self._move_counts = {"singles": 0, "pairs": 0, "five_cards": 0}
        self._final_positions: List[int] = []  # Rankings across games in episode
        self._total_opponent_cards = 0  # For advantage calculation

    def reset_episode(self) -> None:
        """Reset all episode-level tracking at the start of a new episode."""
        self.games_played = 0
        self.games_won = 0
        self.total_cards_when_losing = 0
        self.losses_count = 0
        self._accumulated_move_bonuses = 0.0

        # Reset metrics tracking
        self._episode_steps = 0
        self._move_counts = {"singles": 0, "pairs": 0, "five_cards": 0}
        self._final_positions = []
        self._total_opponent_cards = 0

    def increment_steps(self) -> None:
        """Increment the episode step counter."""
        self._episode_steps += 1

    def add_move_bonus(self, bonus: float) -> None:
        """Add a move bonus to the accumulated episode total."""
        self._accumulated_move_bonuses += bonus

    def track_move_type(self, move_cards: List[int]) -> None:
        """Track the type of move made for strategy metrics."""
        move_count = len(move_cards)
        if move_count == 1:
            self._move_counts["singles"] += 1
        elif move_count == 2:
            self._move_counts["pairs"] += 1
        elif move_count == 5:
            self._move_counts["five_cards"] += 1

    def handle_game_end(self, all_cards_left: List[int]) -> Tuple[int, bool]:
        """Process end-of-game bookkeeping and return winner and controlled player position.

        Args:
            all_cards_left: List of cards remaining for each player

        Returns:
            Tuple of (winner_player_index, controlled_player_won)
        """
        self.games_played += 1

        controlled_cards = all_cards_left[self.controlled_player]

        # Find the actual winner (player with 0 cards)
        winner_player = None
        for i, cards in enumerate(all_cards_left):
            if cards == 0:
                winner_player = i
                break

        # Track final position (1st place = 1, 4th place = 4) based on cards remaining
        sorted_positions = sorted(enumerate(all_cards_left), key=lambda x: x[1])
        position = (
            next(
                i
                for i, (player_idx, _) in enumerate(sorted_positions)
                if player_idx == self.controlled_player
            )
            + 1
        )
        self._final_positions.append(position)

        # Track opponent cards for advantage calculation
        opponent_cards = sum(
            cards
            for i, cards in enumerate(all_cards_left)
            if i != self.controlled_player
        )
        self._total_opponent_cards += opponent_cards

        # Track wins/losses based on whether controlled player won
        controlled_player_won = winner_player == self.controlled_player
        if controlled_player_won:
            self.games_won += 1
        else:
            # Only count cards when losing (not when winning)
            self.total_cards_when_losing += controlled_cards
            self.losses_count += 1

        return winner_player if winner_player is not None else 0, controlled_player_won

    def is_episode_complete(self) -> bool:
        """Check if the current episode is complete."""
        return self.games_played >= self.games_per_episode

    def calculate_episode_bonus(self, reward_function=None) -> float:
        """Calculate episode bonus based on overall performance.

        Args:
            reward_function: Optional reward function with episode_bonus method

        Returns:
            Total episode bonus including accumulated move bonuses
        """
        episode_bonus = 0.0

        if reward_function is not None and self.games_played > 0:
            avg_cards_left = (
                (self.total_cards_when_losing / self.losses_count)
                if self.losses_count > 0
                else 0
            )
            episode_bonus = reward_function.episode_bonus(
                self.games_won, self.games_played, avg_cards_left
            )
        else:
            # Default episode bonus
            win_rate = (
                self.games_won / self.games_played if self.games_played > 0 else 0
            )
            episode_bonus = 0.5 if win_rate > 0.6 else 0

        # Add accumulated move bonuses from complex moves throughout the episode
        total_bonus = episode_bonus + self._accumulated_move_bonuses
        return total_bonus

    def get_episode_metrics(self) -> Dict[str, float]:
        """Get Big Two-specific metrics for the completed episode.

        Returns:
            Dictionary of episode metrics for logging/analysis
        """
        if self.games_played == 0:
            return {}

        # Performance metrics
        win_rate = self.games_won / self.games_played
        avg_cards_remaining = (
            self.total_cards_when_losing / self.losses_count
            if self.losses_count > 0
            else 0
        )
        # Overall average cards left including wins (winner has 0)
        avg_cards_overall = (
            (self.total_cards_when_losing) / self.games_played
            if self.games_played > 0
            else 0
        )
        final_position_avg = (
            sum(self._final_positions) / len(self._final_positions)
            if self._final_positions
            else 0
        )

        # Opponent analysis
        avg_opponent_cards_per_game = self._total_opponent_cards / (
            self.games_played * (self.num_players - 1)
        )
        controlled_cards_per_game = (
            self.total_cards_when_losing / self.losses_count
            if self.losses_count > 0
            else 0
        )
        cards_advantage = avg_opponent_cards_per_game - controlled_cards_per_game

        # Strategy metrics
        total_moves = sum(self._move_counts.values())
        complex_moves = self._move_counts["pairs"] + self._move_counts["five_cards"]
        complex_move_ratio = complex_moves / total_moves if total_moves > 0 else 0
        avg_game_length = self._episode_steps / self.games_played

        # Count dominant wins (wins where average opponent had ≥8 cards)
        dominant_wins = (
            sum(1 for pos in self._final_positions if pos == 1)
            if self.games_won > 0
            else 0
        )

        return {
            "bigtwo/win_rate": win_rate,
            "bigtwo/avg_cards_remaining": avg_cards_remaining,
            "bigtwo/avg_cards_overall": avg_cards_overall,
            "bigtwo/final_position_avg": final_position_avg,
            "bigtwo/cards_advantage": cards_advantage,
            "bigtwo/five_card_hands_played": float(self._move_counts["five_cards"]),
            "bigtwo/complex_move_ratio": complex_move_ratio,
            "bigtwo/move_bonuses_earned": self._accumulated_move_bonuses,
            "bigtwo/avg_game_length": avg_game_length,
            "bigtwo/games_completed": float(self.games_played),
            "bigtwo/games_won": float(self.games_won),
            "bigtwo/games_lost": float(self.losses_count),
        }
