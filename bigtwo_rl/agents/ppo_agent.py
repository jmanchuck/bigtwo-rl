"""PPO agent wrapper for stable-baselines3 models."""

import numpy as np
from stable_baselines3 import PPO
from .base_agent import BaseAgent


class PPOAgent(BaseAgent):
    """PPO agent wrapper for trained models."""

    def __init__(self, model_path=None, model=None, name="PPO"):
        super().__init__(name)
        if model_path:
            self.model = PPO.load(model_path)
        elif model:
            self.model = model
        else:
            raise ValueError("Must provide either model_path or model")

        self.deterministic = True

        # Check if model expects different observation size and needs compatibility
        self.expected_obs_size = self.model.policy.observation_space.shape[0]
        self.needs_compatibility = self.expected_obs_size != 109

    def get_action(self, observation, action_mask=None):
        """Get action from PPO model."""
        # Convert observation if needed for compatibility
        if self.needs_compatibility:
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

    def reset(self):
        """Nothing to reset for PPO agent (stateless)."""
        pass

    def _convert_observation(self, observation):
        """Convert 109-feature observation to model's expected size."""
        if self.expected_obs_size == 107:
            # Model expects 107 features - likely trained without hand_sizes (4) + last_play_exists (1)
            # But had some other features. Try removing the last 2 features (hand_sizes reduced to 2)
            # Current: hand(52) + last_play(52) + hand_sizes(4) + last_play_exists(1) = 109
            # Old likely: hand(52) + last_play(52) + hand_sizes(2) + last_play_exists(1) = 107
            # OR: hand(52) + last_play(52) + hand_sizes(3) = 107

            # Try removing the last 2 features first (most conservative)
            return observation[:107]
        else:
            # For other sizes, warn and truncate/pad
            import warnings

            warnings.warn(
                f"Unsupported model observation size {self.expected_obs_size}, truncating to match"
            )
            if len(observation) > self.expected_obs_size:
                return observation[: self.expected_obs_size]
            else:
                # Pad with zeros
                padded = np.zeros(self.expected_obs_size)
                padded[: len(observation)] = observation
                return padded

    def set_deterministic(self, deterministic: bool):
        """Set whether to use deterministic policy."""
        self.deterministic = deterministic
