"""Big Two RL Environment wrapper with 1,365-action space.

This wrapper provides a clean interface for training RL agents on Big Two with
a fixed action space of exactly 1,365 actions and proper action masking.
"""

import inspect
from typing import Any, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bigtwo_rl.training.rewards.base_reward import BaseReward

from .action import ActionMaskBuilder
from .bigtwo import ToyBigTwoFullRules
from .episode_manager import EpisodeManager


NUM_PLAYERS = 4


class BigTwoWrapper(gym.Env):
    """Gymnasium wrapper for Big Two with 1,365-action space.

    This wrapper provides a clean interface for training RL agents on Big Two
    with optimal performance through:
    - Fixed action space of exactly 1,365 actions
    - Proper action masking using game rules
    - Direct action ID to move translation
    - Multiprocessing-safe lazy initialization
    """

    def __init__(
        self,
        # observation_config: ObservationConfig,
        reward_function: BaseReward,
        num_players: int = 4,
        games_per_episode: int = 10,
        track_move_history: bool = False,
    ):
        """Initialize Big Two RL wrapper.

        Args:
            observation_config: Observation configuration
            reward_function: BaseReward instance for custom rewards
            num_players: Number of players (must be 4)
            games_per_episode: Number of games per training episode
            track_move_history: Whether to track detailed move history

        """
        super().__init__()

        # Validate player count
        if num_players != NUM_PLAYERS:
            raise ValueError("Big Two requires exactly 4 players")

        self.num_players = num_players
        self.games_per_episode = games_per_episode
        self.reward_function = reward_function
        self.track_move_history = track_move_history

        # Store config for lazy initialization (multiprocessing-safe)
        # Fixed action space (breaking change!)
        self.action_space = spaces.Discrete(1365)

        from .action import BitsetFiveCardEngine

        five_engine = BitsetFiveCardEngine()
        self.action_masker = ActionMaskBuilder(five_engine)

        # Initialize observation space immediately (needed for PPO model creation)
        # BasicObservationBuilder uses 168 features (52 + 52 + 64)
        self.observation_space = spaces.Box(low=0, high=1, shape=(168,), dtype=np.float32)

        # Lazy initialization for game components (will be created in reset())
        self.game = None
        self.obs_vectorizer = None
        self.episode_manager = None

        # True self-play experience tracking
        self.player_experiences = []  # List of experiences for each player
        self.current_obs_per_player = {}  # Store observations when they act
        self._current_obs: np.ndarray = None  # Track latest observation

        # Episode tracking
        self.games_completed = 0
        self.episode_complete = False

        # Model reference for true self-play (set by trainer)
        self._model_reference = None

    def _ensure_initialized(self):
        """Ensure game components are initialized (multiprocessing-safe lazy init)."""
        if self.game is None:
            self.game = ToyBigTwoFullRules(self.num_players, self.track_move_history)

        # Observation vectorizer not needed anymore - using direct observation builders
        self.obs_vectorizer = None

        if self.episode_manager is None:
            self.episode_manager = EpisodeManager(
                games_per_episode=self.games_per_episode,
                num_players=self.num_players,
                controlled_player=0,  # All players are "controlled" in self-play
            )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to start a new multi-game episode.

        Args:
            seed: Random seed for reproducibility
            options: Additional options

        Returns:
            Tuple of (initial_observation, info_dict) from current player

        """
        if seed is not None:
            np.random.seed(seed)

        # Ensure components are initialized (multiprocessing-safe)
        self._ensure_initialized()

        # Reset all components
        self.game.reset()
        # Observation builder doesn't need reset
        self.episode_manager.reset_episode()

        # Reset true self-play tracking
        self.player_experiences = []
        self.current_obs_per_player = {}
        self.games_completed = 0
        self.episode_complete = False

        # Get observation from current player's perspective
        obs = self._get_observation(self.game.current_player)
        self._current_obs = obs

        info = {
            "current_player": self.game.current_player,
            "games_completed": self.games_completed,
            "episode_complete": self.episode_complete,
            "legal_actions_count": np.sum(self.get_action_mask()),
        }

        return obs, info

    def step(self, action_id: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute fixed action ID in the game.

        Args:
            action_id: Action ID from 0-1364

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)

        """
        # Ensure initialized
        self._ensure_initialized()

        if self.game.done:
            # Game already finished, check if episode is complete
            return self._handle_episode_completion()

        current_player = self.game.current_player

        # Store current player's observation
        current_obs = self._get_observation(current_player)

        # Validate action is legal
        action_mask = self.get_action_mask()
        if not action_mask[action_id]:
            # Force a legal action (for training stability)
            legal_actions = np.where(action_mask)[0]
            if len(legal_actions) > 0:
                action_id = legal_actions[0]  # Take first legal action
            else:
                raise ValueError("No legal actions available")

        # Translate action to slot indices and execute directly using Hand API
        slot_indices = self._translate_action_to_game_move(action_id, current_player)

        # Execute move using new Hand API - much simpler!
        success = self.game.play_hand_move(current_player, slot_indices)
        if not success:
            # Move was not legal - this shouldn't happen with proper action masking
            # Force pass as fallback
            self.game.play_hand_move(current_player, [])

        # Calculate reward for experience collection (just 0, final rewards applied later)
        player_reward = 0.0

        # Increment step counter for episode metrics
        self.episode_manager.increment_steps()

        # Store experience for training
        self.player_experiences.append(
            {
                "player": current_player,
                "observation": current_obs,
                "action": action_id,
                "reward": player_reward,  # Will be updated with final rewards later
                "done": self.game.done,
                "info": {},
                "legal_moves_count": np.sum(self.get_action_mask()),
            }
        )

        # Handle game/episode completion
        if self.game.done:
            self._apply_final_rewards()
            self.games_completed += 1

            if self.games_completed >= self.games_per_episode:
                self.episode_complete = True
                return self._finalize_episode()
            # Start next game
            self.game.reset()
            # obs_vectorizer is no longer used (using direct observation builders)

        # Return observation for next player
        next_obs = self._get_observation(self.game.current_player)
        self._current_obs = next_obs

        info = {
            "current_player": self.game.current_player,
            "games_completed": self.games_completed,
            "episode_complete": self.episode_complete,
            "action_was_legal": action_mask[action_id],
            "legal_actions_count": np.sum(self.get_action_mask()) if not self.episode_complete else 0,
        }

        return next_obs, player_reward, self.episode_complete, False, info

    # Complex action execution methods removed - using Hand API directly in step()

    def _apply_final_rewards(self):
        """Apply final rewards to all collected experiences when game ends."""
        # Calculate cards left for all players at game end using Hand API
        all_cards_left = self.game.get_player_card_counts()

        # Find winner and process with episode manager
        winner_player, _ = self.episode_manager.handle_game_end(all_cards_left)

        # Apply custom reward function to get final rewards
        final_rewards = []
        if self.reward_function is not None:
            for i in range(self.num_players):
                final_reward = self.reward_function.game_reward(
                    winner_player=winner_player,
                    player_idx=i,
                    cards_left=all_cards_left[i],
                    all_cards_left=all_cards_left,
                )
                final_rewards.append(final_reward)
        else:
            # Default rewards
            for i in range(self.num_players):
                if i == winner_player:
                    final_rewards.append(1.0)  # Winner gets +1
                else:
                    final_rewards.append(
                        -0.1 * all_cards_left[i],
                    )  # Penalty for cards left

        # Update experiences with final rewards
        for exp in self.player_experiences:
            if exp["done"]:  # This experience ended the game
                player_idx = exp["player"]
                exp["reward"] = final_rewards[player_idx]

    def get_action_mask(self) -> np.ndarray:
        """Get 1,365-dimensional legal action mask.

        Returns:
            Boolean array of shape (1365,) where True = legal action

        """
        if self.game.done or self.episode_complete:
            return np.zeros(self.action_space.n, dtype=bool)

        # Use Hand API methods directly - no format conversion needed!
        current_hand = self.game.get_player_hand(self.game.current_player)
        last_played_cards = self.game.get_last_played_cards_encoded()
        is_first_play = self.game.is_first_play()
        has_control = self.game.has_control()

        # Get valid action indices from the action masker
        pass_allowed = not is_first_play  # Can't pass on first play
        valid_action_ids = self.action_masker.full_mask_indices(
            current_hand,
            last_played_cards,
            pass_allowed=pass_allowed,
            is_first_play=is_first_play,
            has_control=has_control,
        )

        # Convert list of valid action IDs to boolean mask
        mask = np.zeros(self.action_space.n, dtype=bool)
        for action_id in valid_action_ids:
            if 0 <= action_id < self.action_space.n:
                mask[action_id] = True

        return mask

    def action_masks(self) -> np.ndarray:
        """Compatibility alias for ActionMasker wrapper."""
        return self.get_action_mask()

    def _get_observation(self, player: int) -> np.ndarray:
        """Get observation for specific player using simplified Hand API.

        Args:
            player: Player index (0-3)

        Returns:
            Observation vector for the player

        """
        from .observation import BasicObservationBuilder

        # Use Hand API methods directly - no format conversion needed!
        player_hand = self.game.get_player_hand(player)
        player_card_counts = self.game.get_player_card_counts()
        last_played_cards = self.game.get_last_played_cards_encoded()
        passes = self.game.passes_in_row
        is_first_play = self.game.is_first_play()

        # Use BasicObservationBuilder for now (can be made configurable later)
        if not hasattr(self, "obs_builder"):
            self.obs_builder = BasicObservationBuilder()

        # Build observation vector
        obs = self.obs_builder.build_observation(
            hand=player_hand,
            current_player=player,
            player_card_counts=player_card_counts,
            last_played_cards=last_played_cards,
            passes=passes,
            is_first_play=is_first_play,
        )

        return obs

    # Move tracking methods removed - simplified architecture

    def _translate_action_to_game_move(self, action_id: int, player_idx: int) -> List[int]:
        """Translate action ID to slot indices for Hand API.

        Args:
            action_id: Action ID from 0-1364
            player_idx: Current player index

        Returns:
            List of slot indices for play_hand_move()

        """
        from .action import OFF_PASS, action_to_tuple

        # Handle pass action
        if action_id == OFF_PASS:
            return []

        # Convert action ID to tuple of card slot indices
        try:
            card_slots = action_to_tuple(action_id)
            return list(card_slots)
        except ValueError:
            # Invalid action ID, fall back to pass
            return []

    # All format conversion methods removed!
    # ToyBigTwoFullRules now provides Hand API methods directly.

    def _handle_episode_completion(self) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Handle when step is called but episode is already complete."""
        return self._finalize_episode()

    def _finalize_episode(self) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Finalize episode with episode bonus and comprehensive metrics."""
        # Calculate episode bonus using episode manager
        episode_bonus = float(
            self.episode_manager.calculate_episode_bonus(self.reward_function),
        )

        # Return dummy observation and signal episode complete
        dummy_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        self._current_obs = dummy_obs

        # Get comprehensive episode metrics for logging
        episode_metrics = self.episode_manager.get_episode_metrics()

        info = {
            "episode_complete": True,
            "games_completed": self.games_completed,
            "episode_bonus": episode_bonus,
            "multi_player_experiences": self.player_experiences,  # All experiences for training
            **episode_metrics,  # Include all Big Two metrics for logging
        }

        return dummy_obs, episode_bonus, True, False, info

    def get_episode_metrics(self) -> dict[str, float]:
        """Get episode metrics from episode manager."""
        if self.episode_manager is None:
            return {}
        return self.episode_manager.get_episode_metrics()

    def render(self, mode: str = "human") -> np.ndarray | None:
        """Render environment (basic game state info)."""
        if mode == "human":
            pass
        return None

    def close(self) -> None:
        """Close environment (no resources to clean up)."""

    @property
    def current_obs(self) -> np.ndarray:
        """Latest observation vector for convenience in diagnostics/tests."""
        if self._current_obs is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return self._current_obs

    # Backward compatibility properties
    @property
    def env(self):
        """Access to underlying game engine for backward compatibility."""
        self._ensure_initialized()
        return self.game

    @property
    def games_played(self) -> int:
        """Total games played in current episode."""
        return self.games_completed

    @property
    def games_won(self) -> int:
        """Number of games won in current episode."""
        if self.episode_manager is None:
            return 0
        return self.episode_manager.games_won

    @property
    def total_cards_when_losing(self) -> int:
        """Total cards remaining when losing games."""
        if self.episode_manager is None:
            return 0
        return self.episode_manager.total_cards_when_losing

    @property
    def losses_count(self) -> int:
        """Number of games lost in current episode."""
        if self.episode_manager is None:
            return 0
        return self.episode_manager.losses_count
