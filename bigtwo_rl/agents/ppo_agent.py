"""PPO agent wrapper for stable-baselines3 models."""

import os
import numpy as np
from typing import Optional, Any, Dict, Tuple, List
from stable_baselines3 import PPO
from .base_agent import BaseAgent
from .model_metadata import ModelMetadata
from ..core.observation_builder import ObservationConfig


class PPOAgent(BaseAgent):
    """PPO agent wrapper for trained models."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        name: str = "PPO",
        observation_config: Optional[Any] = None,
    ) -> None:
        super().__init__(name)
        self.model_path = model_path  # Store for serialization

        # Observation configs
        self.model_obs_config: Optional[ObservationConfig] = None
        # Environment/runtime observation config (may be set later via set_env_reference)
        self.env_obs_config: Optional[ObservationConfig] = None
        if isinstance(observation_config, ObservationConfig):
            self.env_obs_config = observation_config

        if model_path:
            self.model = PPO.load(model_path)

        self.deterministic = True

        # Target size expected by the loaded policy
        self.expected_obs_size = int(self.model.policy.observation_space.shape[0])

        # Prepare feature mapping if we already know the env config
        self._feature_spans_env: Dict[str, Tuple[int, int]] = {}
        self._feature_spans_model: Dict[str, Tuple[int, int]] = {}
        self._feature_names_order = [
            "hand",
            "last_play",
            "hand_sizes",
            "played_cards",
            "remaining_deck",
            "cards_by_player",
            "last_play_exists",
            "game_phase",
            "turn_position",
            "trick_history",
            "pass_history",
            "play_patterns",
            "power_cards_remaining",
            "hand_type_capabilities",
        ]
        self._mapping_ready = False

        # Try to load the model's observation configuration from metadata.json
        self._try_load_model_obs_config()
        if self.env_obs_config is not None and self.model_obs_config is not None:
            self._prepare_feature_mapping()

    def get_action(
        self, observation: np.ndarray, action_mask: Optional[np.ndarray] = None
    ) -> int:
        """Get action from PPO model."""
        # Convert observation to the model's expected feature space if needed
        observation = self._convert_observation(observation)

        # Get model prediction
        action, _ = self.model.predict(observation, deterministic=self.deterministic)

        # Apply action masking manually if the predicted action is invalid
        if action_mask is not None:
            if not action_mask[action]:
                # Find a valid action from the mask
                legal_actions = np.where(action_mask)[0]
                if len(legal_actions) > 0:
                    action = legal_actions[0]  # Take first legal action as fallback
                else:
                    action = 0  # Ultimate fallback

        return int(action)

    def reset(self) -> None:
        """Nothing to reset for PPO agent (stateless)."""
        pass

    def _convert_observation(self, observation: np.ndarray) -> np.ndarray:
        """Convert env observation vector to the model's expected observation space.

        If we have both the model's ObservationConfig and the environment's
        ObservationConfig, perform feature-aware mapping: drop extra features,
        and zero-pad missing ones in the correct order. Otherwise, fall back to
        generic pad/truncate to the model's expected length.
        """
        # Fast path: already the correct size
        if observation.shape[0] == self.expected_obs_size:
            return observation

        if self._mapping_ready:
            # Build output by concatenating per-feature segments in the model's order
            segments: List[np.ndarray] = []
            for feature_name in self._feature_names_order:
                model_span = self._feature_spans_model.get(feature_name)
                if model_span is None or model_span[1] - model_span[0] == 0:
                    # Feature not used by model; skip
                    continue

                target_size = model_span[1] - model_span[0]
                env_span = self._feature_spans_env.get(feature_name)

                if env_span is None or env_span[1] - env_span[0] == 0:
                    # Feature not present in env obs: zero-fill
                    segments.append(np.zeros(target_size, dtype=np.float32))
                else:
                    src = observation[env_span[0] : env_span[1]]
                    # Match sizes conservatively
                    if src.shape[0] == target_size:
                        segments.append(src)
                    elif src.shape[0] > target_size:
                        segments.append(src[:target_size])
                    else:
                        padded = np.zeros(target_size, dtype=np.float32)
                        padded[: src.shape[0]] = src
                        segments.append(padded)

            if segments:
                out = np.concatenate(segments)
                # As a safeguard, adjust to exact expected length
                if out.shape[0] != self.expected_obs_size:
                    if out.shape[0] > self.expected_obs_size:
                        out = out[: self.expected_obs_size]
                    else:
                        padded = np.zeros(self.expected_obs_size, dtype=np.float32)
                        padded[: out.shape[0]] = out
                        out = padded
                return out

        # Fallback: generic pad/truncate
        if observation.shape[0] > self.expected_obs_size:
            return observation[: self.expected_obs_size]
        else:
            padded = np.zeros(self.expected_obs_size, dtype=np.float32)
            padded[: observation.shape[0]] = observation
            return padded

    def set_deterministic(self, deterministic: bool) -> None:
        """Set whether to use deterministic policy."""
        self.deterministic = deterministic

    # --- Environment wiring helpers ---
    def set_env_reference(self, env: Any) -> None:
        """Provide environment reference so we can read its observation config.

        Called by evaluation/tournament code when available.
        """
        env_config = getattr(env, "obs_config", None)
        if isinstance(env_config, ObservationConfig):
            self.env_obs_config = env_config
            if self.model_obs_config is not None:
                self._prepare_feature_mapping()

    # --- Internal helpers ---
    def _try_load_model_obs_config(self) -> None:
        """Load the model's ObservationConfig from metadata.json if available."""
        model_dir = None
        if self.model_path is not None:
            model_dir = os.path.dirname(self.model_path)
        # If created from a PPO model object, there is no path; skip
        if model_dir and os.path.isdir(model_dir):
            config = ModelMetadata.load_observation_config(model_dir)
            if isinstance(config, ObservationConfig):
                self.model_obs_config = config
                # Ensure expected size aligns with policy's observation space
                # but trust the policy space for final length
                # Prepare mapping if env config is already known
                if self.env_obs_config is not None:
                    self._prepare_feature_mapping()

    def _compute_feature_spans(
        self, config: ObservationConfig
    ) -> Dict[str, Tuple[int, int]]:
        """Return feature name -> (start, end) spans for a given config.

        The order must mirror ObservationVectorizer.vectorize.
        """
        sizes: Dict[str, int] = {
            "hand": 52 if config.include_hand else 0,
            "last_play": 52 if config.include_last_play else 0,
            "hand_sizes": 4 if config.include_hand_sizes else 0,
            "played_cards": 52 if config.include_played_cards else 0,
            "remaining_deck": 52 if config.include_remaining_deck else 0,
            "cards_by_player": 208 if config.include_cards_by_player else 0,
            "last_play_exists": 1 if config.include_last_play_exists else 0,
            "game_phase": 3 if config.include_game_phase else 0,
            "turn_position": 4 if config.include_turn_position else 0,
            "trick_history": 12 if config.include_trick_history else 0,
            "pass_history": 4 if config.include_pass_history else 0,
            "play_patterns": 16 if config.include_play_patterns else 0,
            "power_cards_remaining": 5 if config.include_power_cards_remaining else 0,
            "hand_type_capabilities": 20
            if config.include_hand_type_capabilities
            else 0,
        }

        spans: Dict[str, Tuple[int, int]] = {}
        cursor = 0
        for name in self._feature_names_order:
            length = sizes[name]
            spans[name] = (cursor, cursor + length)
            cursor += length
        return spans

    def _prepare_feature_mapping(self) -> None:
        """Precompute feature spans for env and model to enable fast conversion."""
        if self.env_obs_config is None or self.model_obs_config is None:
            self._mapping_ready = False
            return
        # Ensure internal sizes are up to date
        self.env_obs_config.__post_init__()
        self.model_obs_config.__post_init__()
        self._feature_spans_env = self._compute_feature_spans(self.env_obs_config)
        self._feature_spans_model = self._compute_feature_spans(self.model_obs_config)
        self._mapping_ready = True
