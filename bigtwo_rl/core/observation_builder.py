"""Configurable observation space builder for Big Two RL training.

This module allows users to customize exactly what information their models see,
enabling experiments with different levels of game state visibility.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from gymnasium import spaces


@dataclass
class ObservationConfig:
    """Configuration for what information the agent can observe."""

    # Core game state (always needed)
    include_hand: bool = True  # Own cards (52 features)
    include_last_play: bool = True  # Last played cards (52 features)
    include_hand_sizes: bool = True  # All players' card counts (4 features)

    # Enhanced memory features
    include_played_cards: bool = False  # All cards played so far (52 features)
    include_remaining_deck: bool = False  # What cards are still unplayed (52 features)
    include_cards_by_player: bool = False  # Which player played which cards (52*4=208 features)

    # Game context features
    include_last_play_exists: bool = True  # Whether there's a last play (1 feature)
    include_game_phase: bool = False  # Early/mid/late game indicator (3 features)
    include_turn_position: bool = False  # Position relative to dealer (4 features)
    include_trick_history: bool = False  # Who won last N tricks (12 features, last 3 tricks * 4 players)

    # Opponent modeling features
    include_pass_history: bool = False  # Who passed on current trick (4 features)
    include_play_patterns: bool = False  # Opponent playing style indicators (16 features, 4 players * 4 style metrics)

    # Advanced strategic features
    include_power_cards_remaining: bool = False  # Status of A♠, 2♠, 2♥, 2♦, 2♣ (5 features)
    include_hand_type_capabilities: bool = (
        False  # What hand types each player can still make (20 features, 4 players * 5 types)
    )

    # Internal tracking (computed automatically)
    _feature_sizes: dict[str, int] = field(default_factory=dict, init=False)
    _total_size: int = field(default=0, init=False)

    def __post_init__(self):
        """Calculate feature sizes and total observation space size."""
        self._feature_sizes = {
            "hand": 52 if self.include_hand else 0,
            "last_play": 52 if self.include_last_play else 0,
            "hand_sizes": 4 if self.include_hand_sizes else 0,
            "played_cards": 52 if self.include_played_cards else 0,
            "remaining_deck": 52 if self.include_remaining_deck else 0,
            "cards_by_player": 208 if self.include_cards_by_player else 0,
            "last_play_exists": 1 if self.include_last_play_exists else 0,
            "game_phase": 3 if self.include_game_phase else 0,
            "turn_position": 4 if self.include_turn_position else 0,
            "trick_history": 12 if self.include_trick_history else 0,
            "pass_history": 4 if self.include_pass_history else 0,
            "play_patterns": 16 if self.include_play_patterns else 0,
            "power_cards_remaining": 5 if self.include_power_cards_remaining else 0,
            "hand_type_capabilities": 20 if self.include_hand_type_capabilities else 0,
        }
        self._total_size = sum(self._feature_sizes.values())

        # Validation
        if not (self.include_hand and self.include_hand_sizes):
            raise ValueError(
                "include_hand and include_hand_sizes are required for basic gameplay",
            )


class ObservationBuilder:
    """Builder class for creating custom observation configurations."""

    def __init__(self):
        """Initialize with minimal required features."""
        self.config = ObservationConfig(
            include_hand=True,
            include_hand_sizes=True,
            include_last_play=False,
            include_last_play_exists=False,
        )

    # Core features (fluent interface)
    def with_last_play(self, enabled: bool = True) -> "ObservationBuilder":
        """Include information about the last played cards."""
        self.config.include_last_play = enabled
        self.config.include_last_play_exists = enabled  # Usually want both
        return self

    def with_card_memory(self, enabled: bool = True) -> "ObservationBuilder":
        """Include memory of all cards played so far."""
        self.config.include_played_cards = enabled
        return self

    def with_remaining_deck(self, enabled: bool = True) -> "ObservationBuilder":
        """Include information about what cards are still unplayed."""
        self.config.include_remaining_deck = enabled
        return self

    def with_full_card_tracking(self, enabled: bool = True) -> "ObservationBuilder":
        """Include detailed tracking of which player played which cards."""
        self.config.include_cards_by_player = enabled
        return self

    # Game context features
    def with_game_context(self, enabled: bool = True) -> "ObservationBuilder":
        """Include game phase and turn position information."""
        self.config.include_game_phase = enabled
        self.config.include_turn_position = enabled
        return self

    def with_trick_history(self, enabled: bool = True) -> "ObservationBuilder":
        """Include information about who won recent tricks."""
        self.config.include_trick_history = enabled
        return self

    # Strategic features
    def with_power_card_tracking(self, enabled: bool = True) -> "ObservationBuilder":
        """Track status of the most powerful cards (Aces and 2s)."""
        self.config.include_power_cards_remaining = enabled
        return self

    def with_opponent_modeling(self, enabled: bool = True) -> "ObservationBuilder":
        """Include features for modeling opponent behavior."""
        self.config.include_pass_history = enabled
        self.config.include_play_patterns = enabled
        return self

    def with_hand_type_analysis(self, enabled: bool = True) -> "ObservationBuilder":
        """Include analysis of what hand types players can still form."""
        self.config.include_hand_type_capabilities = enabled
        return self

    def standard(self) -> "ObservationBuilder":
        """Standard observation: current state only (backward compatible)."""
        self.config = ObservationConfig(
            include_hand=True,
            include_last_play=True,
            include_hand_sizes=True,
            include_last_play_exists=True,
        )
        return self

    def memory_enhanced(self) -> "ObservationBuilder":
        """Enhanced with card memory and game context."""
        return self.standard().with_card_memory().with_remaining_deck().with_game_context()

    def strategic(self) -> "ObservationBuilder":
        """Full strategic information including opponent modeling."""
        return (
            self.memory_enhanced()
            .with_full_card_tracking()
            .with_trick_history()
            .with_power_card_tracking()
            .with_opponent_modeling()
            .with_hand_type_analysis()
        )

    def build(self) -> ObservationConfig:
        """Build and return the final observation configuration."""
        # Recalculate sizes after all modifications
        self.config.__post_init__()
        return self.config


class ObservationVectorizer:
    """Converts game state to observation vector based on configuration."""

    def __init__(self, config: ObservationConfig):
        self.config = config
        self.gymnasium_space = spaces.Box(
            low=-1,
            high=1,
            shape=(config._total_size,),
            dtype=np.float32,
        )

        # Initialize tracking state for advanced features
        self._played_cards = np.zeros(52, dtype=bool)  # Cards played this game
        self._cards_by_player = np.zeros((4, 52), dtype=bool)  # Who played what
        self._trick_winners = []  # Last few trick winners
        self._pass_counts = np.zeros(4, dtype=int)  # Pass counts per player
        self._play_style_metrics = np.zeros(
            (4, 4),
            dtype=float,
        )  # Style indicators per player

    def reset(self):
        """Reset tracking state for new game."""
        self._played_cards.fill(False)
        self._cards_by_player.fill(False)
        self._trick_winners.clear()
        self._pass_counts.fill(0)
        self._play_style_metrics.fill(0)

    def vectorize(self, raw_obs: dict[str, Any], game_env) -> np.ndarray:
        """Convert raw observation dict to configured feature vector."""
        features = []

        # Collect features from different categories
        features.extend(self._build_core_features(raw_obs, game_env))
        features.extend(self._build_memory_features(game_env))
        features.extend(self._build_context_features(raw_obs, game_env))
        features.extend(self._build_opponent_modeling_features())
        features.extend(self._build_strategic_features())

        # Concatenate all features
        if not features:
            return np.array([], dtype=np.float32)

        result = np.concatenate(features)

        # Ensure correct size
        if len(result) != self.config._total_size:
            raise ValueError(
                f"Feature vector size mismatch: got {len(result)}, expected {self.config._total_size}",
            )

        return result

    def _build_core_features(
        self,
        raw_obs: dict[str, Any],
        game_env,
    ) -> list[np.ndarray]:
        """Build core game state features (hand, last_play, hand_sizes)."""
        features = []

        if self.config.include_hand:
            features.append(raw_obs["hand"].astype(np.float32))

        if self.config.include_last_play:
            features.append(raw_obs["last_play"].astype(np.float32))

        if self.config.include_hand_sizes:
            hand_sizes = np.zeros(4, dtype=np.float32)
            hand_sizes[: game_env.num_players] = np.sum(
                game_env.hands,
                axis=1,
                dtype=np.float32,
            )
            features.append(hand_sizes)

        return features

    def _build_memory_features(self, game_env) -> list[np.ndarray]:
        """Build memory-based features (played_cards, remaining_deck, cards_by_player)."""
        features = []

        if self.config.include_played_cards:
            features.append(self._played_cards.astype(np.float32))

        if self.config.include_remaining_deck:
            remaining = self._calculate_remaining_deck(game_env)
            features.append(remaining.astype(np.float32))

        if self.config.include_cards_by_player:
            features.append(self._cards_by_player.flatten().astype(np.float32))

        return features

    def _build_context_features(
        self,
        raw_obs: dict[str, Any],
        game_env,
    ) -> list[np.ndarray]:
        """Build context features (last_play_exists, game_phase, turn_position, trick_history)."""
        features = []

        if self.config.include_last_play_exists:
            features.append(np.array([raw_obs["last_play_exists"]], dtype=np.float32))

        if self.config.include_game_phase:
            phase = self._calculate_game_phase(game_env)
            features.append(np.array(phase, dtype=np.float32))

        if self.config.include_turn_position:
            position = np.zeros(4, dtype=np.float32)
            position[game_env.current_player] = 1
            features.append(position)

        if self.config.include_trick_history:
            history = self._calculate_trick_history()
            features.append(history)

        return features

    def _build_opponent_modeling_features(self) -> list[np.ndarray]:
        """Build opponent modeling features (pass_history, play_patterns)."""
        features = []

        if self.config.include_pass_history:
            features.append(self._pass_counts.astype(np.float32))

        if self.config.include_play_patterns:
            features.append(self._play_style_metrics.flatten().astype(np.float32))

        return features

    def _build_strategic_features(self) -> list[np.ndarray]:
        """Build strategic features (power_cards_remaining, hand_type_capabilities)."""
        features = []

        if self.config.include_power_cards_remaining:
            power_remaining = self._calculate_power_cards_status()
            features.append(power_remaining)

        if self.config.include_hand_type_capabilities:
            # Placeholder: analyze what hand types each player could form
            capabilities = np.zeros(20, dtype=np.float32)  # 4 players * 5 hand types
            features.append(capabilities)

        return features

    def _calculate_remaining_deck(self, game_env) -> np.ndarray:
        """Calculate which cards are still unplayed and not in any hand."""
        remaining = np.ones(52, dtype=bool)
        for player_hand in game_env.hands:
            remaining &= ~player_hand
        remaining &= ~self._played_cards
        return remaining

    def _calculate_game_phase(self, game_env) -> list[int]:
        """Calculate current game phase (early/mid/late)."""
        total_cards = np.sum(game_env.hands)
        if total_cards > 40:  # Early game
            return [1, 0, 0]
        if total_cards > 20:  # Mid game
            return [0, 1, 0]
        # Late game
        return [0, 0, 1]

    def _calculate_trick_history(self) -> np.ndarray:
        """Calculate last 3 trick winners as one-hot encoded features."""
        history = np.zeros(12, dtype=np.float32)
        for i, winner in enumerate(self._trick_winners[-3:]):
            if winner < 4:  # Valid player
                history[i * 4 + winner] = 1
        return history

    def _calculate_power_cards_status(self) -> np.ndarray:
        """Calculate status of power cards (A♠, 2♠, 2♥, 2♦, 2♣)."""
        power_card_indices = [51, 47, 35, 23, 11]  # Approximate indices for these cards
        power_remaining = np.zeros(5, dtype=np.float32)
        for i, idx in enumerate(power_card_indices):
            if idx < 52 and not self._played_cards[idx]:
                power_remaining[i] = 1
        return power_remaining

    def update_tracking(self, move, player, game_env):
        """Update tracking state when a move is played."""
        # For numpy arrays, len(move) is 52 regardless of pass/non-pass; use np.any
        is_numpy_array = isinstance(move, np.ndarray)
        is_pass = False
        if move is None:
            is_pass = True
        elif is_numpy_array:
            is_pass = not np.any(move)
        else:
            # For list-like, empty list means pass
            try:
                is_pass = len(move) == 0
            except Exception:
                is_pass = False

        # Update played cards for non-pass numpy array moves
        if is_numpy_array and not is_pass:
            self._played_cards |= move
            self._cards_by_player[player] |= move

        # Update pass tracking
        if is_pass:
            self._pass_counts[player] += 1


def strategic_observation() -> ObservationConfig:
    """Full strategic observation configuration."""
    return ObservationBuilder().strategic().build()
