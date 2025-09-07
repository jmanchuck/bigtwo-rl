"""Game state extraction utilities for observation builders."""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

from ..bigtwo import ToyBigTwoFullRules
from ..game.types import Hand


@dataclass
class GameState:
    """Extracted game state data for observation builders."""

    current_player: int
    player_hands: List[Hand]
    player_card_counts: List[int]
    last_played_cards: List[int]
    passes: int
    is_first_play: bool
    has_control: bool


class GameStateExtractor:
    """Utility class to extract standardized game state from ToyBigTwoFullRules game objects."""

    @staticmethod
    def extract_game_state(game: ToyBigTwoFullRules, player_perspective: int = 0) -> GameState:
        """
        Extract game state from ToyBigTwoFullRules game object.

        Args:
            game: ToyBigTwoFullRules game instance
            player_perspective: Index of player whose perspective to extract (0-3)

        Returns:
            GameState object with extracted information
        """
        return GameState(
            current_player=game.current_player,
            player_hands=[game.get_player_hand(i) for i in range(game.num_players)],
            player_card_counts=game.get_player_card_counts(),
            last_played_cards=game.get_last_played_cards_encoded(),
            passes=game.passes_in_row,
            is_first_play=game.is_first_play(),
            has_control=game.has_control(),
        )

    @staticmethod
    def get_player_observation_data(game: ToyBigTwoFullRules, player_idx: int) -> Dict[str, Any]:
        """
        Get observation data for a specific player.

        Args:
            game: ToyBigTwoFullRules game instance
            player_idx: Index of player (0-3)

        Returns:
            Dictionary with data needed for observation builders
        """
        game_state = GameStateExtractor.extract_game_state(game, player_idx)

        return {
            "hand": game_state.player_hands[player_idx],
            "current_player": game_state.current_player,
            "player_card_counts": game_state.player_card_counts,
            "last_played_cards": game_state.last_played_cards,
            "passes": game_state.passes,
            "is_first_play": game_state.is_first_play,
        }

    @staticmethod
    def get_relative_perspective(game: ToyBigTwoFullRules, player_idx: int) -> Dict[str, Any]:
        """
        Get game state from a specific player's relative perspective.

        Args:
            game: ToyBigTwoFullRules game instance
            player_idx: Index of player whose perspective to use

        Returns:
            Dictionary with relative game state information
        """
        game_state = GameStateExtractor.extract_game_state(game)

        # Rotate player indices so current player is always 0
        def rotate_player_idx(idx: int) -> int:
            return (idx - player_idx) % 4

        # Reorder card counts relative to current player
        relative_card_counts = [game_state.player_card_counts[(player_idx + i) % 4] for i in range(4)]

        return {
            "hand": game_state.player_hands[player_idx],
            "current_player": rotate_player_idx(game_state.current_player),
            "player_card_counts": relative_card_counts,
            "last_played_cards": game_state.last_played_cards,
            "passes": game_state.passes,
            "is_first_play": game_state.is_first_play,
            "has_control": game_state.has_control,
            "original_player_idx": player_idx,
        }


class ObservationOrchestrator:
    """Orchestrator to manage observation generation for different players."""

    def __init__(self, observation_builder):
        """
        Initialize with an observation builder instance.

        Args:
            observation_builder: Instance of ObservationBuilder or subclass
        """
        self.observation_builder = observation_builder
        self.extractor = GameStateExtractor()

    def get_observation(self, game: ToyBigTwoFullRules, player_idx: int):
        """
        Get observation for a specific player.

        Args:
            game: ToyBigTwoFullRules game instance
            player_idx: Index of player (0-3)

        Returns:
            Observation vector for the specified player
        """
        obs_data = self.extractor.get_player_observation_data(game, player_idx)

        return self.observation_builder.build_observation(
            hand=obs_data["hand"],
            current_player=obs_data["current_player"],
            player_card_counts=obs_data["player_card_counts"],
            last_played_cards=obs_data["last_played_cards"],
            passes=obs_data["passes"],
            is_first_play=obs_data["is_first_play"],
        )

    def get_relative_observation(self, game: ToyBigTwoFullRules, player_idx: int):
        """
        Get observation from a specific player's relative perspective.

        Args:
            game: ToyBigTwoFullRules game instance
            player_idx: Index of player whose perspective to use

        Returns:
            Observation vector from player's relative perspective
        """
        obs_data = self.extractor.get_relative_perspective(game, player_idx)

        return self.observation_builder.build_observation(
            hand=obs_data["hand"],
            current_player=obs_data["current_player"],
            player_card_counts=obs_data["player_card_counts"],
            last_played_cards=obs_data["last_played_cards"],
            passes=obs_data["passes"],
            is_first_play=obs_data["is_first_play"],
        )

    def get_all_player_observations(self, game: ToyBigTwoFullRules):
        """
        Get observations for all players.

        Args:
            game: ToyBigTwoFullRules game instance

        Returns:
            List of observation vectors, one for each player
        """
        return [self.get_observation(game, player_idx) for player_idx in range(4)]

    def get_observation_info(self) -> Dict[str, Any]:
        """Get information about the observation builder being used."""
        return {
            "builder_type": type(self.observation_builder).__name__,
            "observation_size": self.observation_builder.observation_size,
            "feature_info": self.observation_builder.get_feature_info(),
        }
