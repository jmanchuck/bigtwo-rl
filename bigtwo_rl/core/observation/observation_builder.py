"""Abstract base class for observation builders."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

from ..game.types import Hand


class ObservationBuilder(ABC):
    """Abstract interface for building observation vectors from Big Two game state."""

    def __init__(self, observation_size: int):
        """
        Initialize observation builder.

        Args:
            observation_size: Size of the output observation vector
        """
        self.observation_size = observation_size

    @abstractmethod
    def encode_hand(self, hand: Hand) -> np.ndarray:
        """
        Encode a player's hand into feature vector.

        Args:
            hand: Player's current hand

        Returns:
            Feature vector representing the hand
        """
        pass

    @abstractmethod
    def encode_last_play(self, last_played_cards: List[int], passes: int) -> np.ndarray:
        """
        Encode the last play

        Args:
            last_played_cards: List of encoded cards from last play (empty if no last play)
            passes: Number of consecutive passes

        Returns:
            Feature vector representing the last play
        """
        pass

    @abstractmethod
    def encode_game_state(
        self, current_player: int, player_card_counts: List[int], passes: int, is_first_play: bool
    ) -> np.ndarray:
        """
        Encode general game state information.

        Args:
            current_player: Index of current player (0-3)
            player_card_counts: Number of cards each player has
            passes: Number of consecutive passes
            is_first_play: Whether this is the first play of the game

        Returns:
            Feature vector representing game state
        """
        pass

    def build_observation(
        self,
        hand: Hand,
        current_player: int,
        player_card_counts: List[int],
        last_played_cards: List[int],
        passes: int,
        is_first_play: bool,
    ) -> np.ndarray:
        """
        Build complete observation vector from all game state components.

        Args:
            hand: Current player's hand
            current_player: Current player index
            player_card_counts: Card counts for all players
            last_played_cards: List of encoded cards from last play
            passes: Number of consecutive passes
            is_first_play: Whether this is the first play of the game

        Returns:
            Complete observation vector of size self.observation_size
        """
        return np.concatenate(
            [
                self.encode_hand(hand),
                self.encode_last_play(last_played_cards, passes),
                self.encode_game_state(current_player, player_card_counts, passes, is_first_play),
            ]
        )

    @abstractmethod
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about the observation features for debugging/analysis.

        Returns:
            Dictionary describing the observation structure and feature meanings
        """
        pass
