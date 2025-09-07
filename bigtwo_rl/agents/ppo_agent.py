"""PPO agent for fixed 1,365-action space Big Two."""

import numpy as np
from typing import Optional, Any
from pathlib import Path

from .base_agent import BaseAgent

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy

    SB3_AVAILABLE = True
except ImportError:
    # Explicitly annotate to avoid shadowing type checker errors
    PPO: Any | None = None
    ActorCriticPolicy: Any | None = None
    SB3_AVAILABLE = False


class PPOAgent(BaseAgent):
    """PPO agent for Big Two with fixed 1,365-action space.

    This agent uses trained PPO models from stable-baselines3 to play Big Two.
    Requires models trained specifically for the 1,365-action space.
    """

    def __init__(self, model_path: str, name: str = "PPOAgent", deterministic: bool = True):
        """Initialize PPO agent with trained model.

        Args:
            model_path: Path to trained PPO model (.zip file)
            name: Agent name for identification
            deterministic: Whether to use deterministic policy (exploitation)
        """
        super().__init__(name)

        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for PPOAgent. Install with: pip install stable-baselines3")

        self.model_path = Path(model_path)
        self.deterministic = deterministic
        # Annotate as Any to satisfy type checker when SB3 not installed
        self.model: Any = None

        # Load model
        self._load_model()

    def _load_model(self) -> None:
        """Load the PPO model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        try:
            self.model = PPO.load(self.model_path)
            print(f"✓ Loaded PPO model from {self.model_path}")

            # Verify action space
            expected_action_space = 1365
            if hasattr(self.model, "action_space"):
                if self.model.action_space.n != expected_action_space:
                    print(
                        f"⚠️  Warning: Model action space is {self.model.action_space.n}, "
                        f"expected {expected_action_space}"
                    )

        except Exception as e:
            raise RuntimeError(f"Failed to load PPO model: {e}")

    def get_action(self, observation: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        """Get action from PPO policy with action masking.

        Args:
            observation: Game observation vector (168 features)
            action_mask: 1365-dim boolean mask for legal actions

        Returns:
            Action ID from 0-1364
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Apply action mask if provided
        if action_mask is not None:
            # Create masked action space
            legal_actions = np.where(action_mask)[0]
            if len(legal_actions) == 0:
                # No legal actions - shouldn't happen, but return 0 as fallback
                print("Warning: No legal actions available, returning action 0")
                return 0

            # Get action probabilities from policy
            try:
                # Use the policy to get action probabilities
                obs_tensor = self.model.policy.obs_to_tensor(observation.reshape(1, -1))[0]

                with self.model.policy.set_training_mode(False):
                    distribution = self.model.policy.get_distribution(obs_tensor)

                    if self.deterministic:
                        # Get the most likely action among legal actions
                        action_logits = distribution.distribution.logits.detach().cpu().numpy()[0]

                        # Mask out illegal actions by setting their logits to -inf
                        masked_logits = np.full(1365, -np.inf)
                        masked_logits[legal_actions] = action_logits[legal_actions]

                        # Select action with highest masked logit
                        action = int(np.argmax(masked_logits))
                    else:
                        # Sample from masked distribution
                        action_probs = distribution.distribution.probs.detach().cpu().numpy()[0]

                        # Create masked probability distribution
                        masked_probs = np.zeros(1365)
                        masked_probs[legal_actions] = action_probs[legal_actions]

                        # Renormalize
                        if np.sum(masked_probs) > 0:
                            masked_probs = masked_probs / np.sum(masked_probs)
                            action = int(np.random.choice(1365, p=masked_probs))
                        else:
                            # Fallback to uniform random among legal actions
                            action = int(np.random.choice(legal_actions))

                return action

            except Exception as e:
                print(f"Warning: PPO policy evaluation failed ({e}), falling back to random legal action")
                return int(np.random.choice(legal_actions))

        else:
            # No action mask - use policy directly (may select illegal actions)
            action, _ = self.model.predict(observation, deterministic=self.deterministic)
            return int(action)

    def reset(self) -> None:
        """Reset agent state.

        PPO agent has no internal state to reset.
        """
        pass

    def get_action_distribution(self, observation: np.ndarray, action_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Get action probability distribution from PPO policy.

        Args:
            observation: Game observation vector
            action_mask: Legal action mask

        Returns:
            Probability distribution over actions
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            # Get policy distribution
            obs_tensor = self.model.policy.obs_to_tensor(observation.reshape(1, -1))[0]

            with self.model.policy.set_training_mode(False):
                distribution = self.model.policy.get_distribution(obs_tensor)
                action_probs = distribution.distribution.probs.detach().cpu().numpy()[0]

            # Apply action mask if provided
            if action_mask is not None:
                legal_actions = np.where(action_mask)[0]
                masked_probs = np.zeros(1365)

                if len(legal_actions) > 0:
                    masked_probs[legal_actions] = action_probs[legal_actions]

                    # Renormalize
                    if np.sum(masked_probs) > 0:
                        masked_probs = masked_probs / np.sum(masked_probs)
                    else:
                        # Uniform over legal actions as fallback
                        masked_probs[legal_actions] = 1.0 / len(legal_actions)

                return masked_probs

            return action_probs

        except Exception as e:
            print(f"Warning: Failed to get action distribution ({e})")

            # Fallback to uniform distribution
            if action_mask is not None:
                legal_actions = np.where(action_mask)[0]
                uniform_probs = np.zeros(1365)
                if len(legal_actions) > 0:
                    uniform_probs[legal_actions] = 1.0 / len(legal_actions)
                return uniform_probs
            else:
                return np.ones(1365) / 1365


# Convenience function for creating PPO agents
def load_ppo_agent(model_path: str, name: Optional[str] = None, deterministic: bool = True) -> PPOAgent:
    """Load a trained PPO agent from file.

    Args:
        model_path: Path to trained model (.zip file)
        name: Agent name (defaults to filename)
        deterministic: Whether to use deterministic policy

    Returns:
        Loaded PPOAgent instance
    """
    if name is None:
        name = Path(model_path).stem

    return PPOAgent(model_path, name, deterministic)
