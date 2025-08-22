"""Game state tracking and caching for Big Two RL environments."""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .bigtwo import ToyBigTwoFullRules


class GameStateTracker:
    """Manages game state tracking, caching, and action mask generation for Big Two.

    Handles legal move caching, game state hashing for performance optimization,
    and action mask generation while keeping this logic separate from the core RL wrapper.
    """

    def __init__(self, action_space_size: int = 2000):
        """Initialize game state tracker.

        Args:
            action_space_size: Size of the action space for mask generation
        """
        self.action_space_size = action_space_size

        # Cache for legal moves to avoid duplicate computation
        self._cached_legal_moves: Optional[List[np.ndarray]] = None
        self._cache_player: Optional[int] = None
        self._cache_state_hash: Optional[Tuple] = None
        self._cache_turn_counter: int = 0

    def reset_cache(self) -> None:
        """Reset all caching state."""
        self._cached_legal_moves = None
        self._cache_player = None
        self._cache_state_hash = None
        self._cache_turn_counter = 0

    def increment_turn_counter(self) -> None:
        """Increment the turn counter for cache invalidation."""
        self._cache_turn_counter += 1

    def get_simple_state_hash(
        self, game_env: "ToyBigTwoFullRules"
    ) -> Tuple[int, Tuple[int, ...], int, int]:
        """Create a lightweight hash of game state for cache invalidation.

        Args:
            game_env: The Big Two game environment

        Returns:
            Tuple representing the current game state for caching
        """
        # Use turn counter and hand sizes instead of expensive tuple creation
        hand_sizes = tuple(np.sum(game_env.hands, axis=1))
        last_play_len = np.sum(game_env.last_play[0]) if game_env.last_play else 0
        return (
            game_env.current_player,
            hand_sizes,
            last_play_len,
            self._cache_turn_counter,
        )

    def get_legal_moves_cached(
        self, game_env: "ToyBigTwoFullRules", player: int
    ) -> List[np.ndarray]:
        """Get legal moves with optimized caching to avoid duplicate computation.

        Args:
            game_env: The Big Two game environment
            player: Player index to get legal moves for

        Returns:
            List of legal moves as numpy arrays
        """
        current_state = self.get_simple_state_hash(game_env)

        # Check if cache is valid
        if (
            self._cache_player == player
            and self._cache_state_hash == current_state
            and self._cached_legal_moves is not None
        ):
            return self._cached_legal_moves

        # Cache miss - compute and store
        self._cached_legal_moves = game_env.legal_moves(player)
        self._cache_player = player
        self._cache_state_hash = current_state

        return self._cached_legal_moves

    def get_action_mask(
        self, game_env: "ToyBigTwoFullRules", current_player: int
    ) -> np.ndarray:
        """Return boolean mask for legal actions.

        Args:
            game_env: The Big Two game environment
            current_player: Current player index

        Returns:
            Boolean array indicating which actions are legal
        """
        legal_moves = self.get_legal_moves_cached(game_env, current_player)
        mask = np.zeros(self.action_space_size, dtype=bool)

        for i, move in enumerate(legal_moves):
            if i < self.action_space_size:  # Safety check
                mask[i] = True

        return mask

    def find_pass_move_idx(self, legal_moves: List[np.ndarray]) -> Optional[int]:
        """Find index of pass move (all-False array or empty list) in legal_moves list.

        Args:
            legal_moves: List of legal moves to search through

        Returns:
            Index of pass move if found, None otherwise
        """
        for i, move in enumerate(legal_moves):
            # Check for numpy array pass move (all False)
            if isinstance(move, np.ndarray):
                if not np.any(move):
                    return i
            # Check for list pass move (empty list)
            elif isinstance(move, list) and len(move) == 0:
                return i
        return None

    def validate_and_clamp_action(
        self, action: int, legal_moves: List[np.ndarray]
    ) -> int:
        """Validate an action and clamp to legal range if needed.

        Args:
            action: Raw action index
            legal_moves: List of currently legal moves

        Returns:
            Valid action index within legal moves range
        """
        if len(legal_moves) == 0:
            return 0

        # Map oversized actions into current legal move range to avoid systematic passes
        return int(action) % len(legal_moves)

    def clear_cache_for_new_game(self) -> None:
        """Clear caches when starting a new game."""
        self._cached_legal_moves = None
        self._cache_player = None
        self._cache_state_hash = None
        self._cache_turn_counter = 0
