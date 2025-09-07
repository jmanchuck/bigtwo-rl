"""Training pipeline for Big Two agents with 1,365-action space."""

import os
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Any as TypingAny

import numpy as np
import torch

from .rewards.base_reward import BaseReward

# Try to import stable-baselines3 components
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
    from stable_baselines3.common.logger import configure

    SB3_AVAILABLE = True
except ImportError:
    # Explicitly annotate Nones to avoid shadowing issues in type checker
    PPO: TypingAny | None = None
    make_vec_env: TypingAny | None = None
    ActorCriticPolicy: TypingAny | None = None
    EvalCallback: TypingAny | None = None
    BaseCallback: TypingAny | None = None
    configure: TypingAny | None = None
    SB3_AVAILABLE = False


class MaskedActorCriticPolicy(ActorCriticPolicy):
    """Actor-Critic policy with action masking support."""

    def forward(self, obs, deterministic=False, action_masks=None):
        """Forward pass with optional action masking.

        Args:
            obs: Observations
            deterministic: Whether to use deterministic actions
            action_masks: Boolean masks for legal actions

        Returns:
            actions, values, log_probs
        """
        # Get action distribution from policy
        distribution = self._get_action_dist_from_latent(self._get_latent(obs)[0])

        # Apply action masks if provided
        if action_masks is not None:
            # Mask out illegal actions by setting their logits to -inf
            logits = distribution.distribution.logits
            masked_logits = torch.where(
                action_masks, logits, torch.tensor(-float("inf"), device=logits.device, dtype=logits.dtype)
            )

            # Create new distribution with masked logits
            from torch.distributions import Categorical

            distribution.distribution = Categorical(logits=masked_logits)

        actions = distribution.get_actions(deterministic=deterministic)
        values = self.value_net(self._get_latent(obs)[1])
        log_probs = distribution.log_prob(actions)

        return actions, values, log_probs.reshape(-1, 1)


class ActionMaskingWrapper:
    """Wrapper to provide action masking for the environment."""

    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        return self.env.close()

    def get_action_mask(self):
        """Get action mask from environment."""
        return self.env.get_action_mask()


class Trainer:
    """Training pipeline for Big Two agents."""

    def __init__(
        self,
        reward_function: Optional[BaseReward] = None,
        num_players: int = 4,
        games_per_episode: int = 5,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        n_steps: int = 512,
        batch_size: int = 64,
        n_epochs: int = 10,
        clip_range: float = 0.2,
        device: str = "auto",
    ):
        """Initialize trainer.

        Args:
            reward_function: Reward function to use
            num_players: Number of players (must be 4)
            games_per_episode: Games per episode
            learning_rate: Learning rate for PPO
            gamma: Discount factor
            n_steps: Steps per rollout
            batch_size: Batch size for updates
            n_epochs: Epochs per update
            clip_range: PPO clipping range
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for training. Install with: pip install stable-baselines3")

        # Import DefaultReward here to avoid circular import
        if reward_function is None:
            from .rewards import DefaultReward

            reward_function = DefaultReward()
        self.reward_function = reward_function
        self.num_players = num_players
        self.games_per_episode = games_per_episode

        # PPO hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.device = device

        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"✓ Trainer initialized with device: {self.device}")

    def _create_env(self):
        """Create training environment."""
        # Import here to avoid circular import
        from ..core.bigtwo_wrapper import BigTwoWrapper

        return BigTwoWrapper(
            reward_function=self.reward_function, num_players=self.num_players, games_per_episode=self.games_per_episode
        )

    def train(
        self,
        total_timesteps: int = 25000,
        model_name: Optional[str] = None,
        log_dir: str = "./logs",
        save_dir: str = "./models",
        eval_freq: int = 5000,
        verbose: int = 1,
    ) -> Tuple[PPO, str]:
        """Train a PPO agent.

        Args:
            total_timesteps: Total training timesteps
            model_name: Name for the model (defaults to timestamp)
            log_dir: Directory for logs
            save_dir: Directory to save models
            eval_freq: Evaluation frequency
            verbose: Verbosity level

        Returns:
            Tuple of (trained_model, model_directory)
        """
        # Create model name if not provided
        if model_name is None:
            model_name = f"bigtwo_ppo_{int(time.time())}"

        # Create directories
        model_dir = Path(save_dir) / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        log_path = Path(log_dir) / model_name
        log_path.mkdir(parents=True, exist_ok=True)

        print(f"Training model: {model_name}")
        print(f"Model directory: {model_dir}")
        print(f"Log directory: {log_path}")

        # Create environment
        env = self._create_env()

        # Create policy with action masking support
        policy_kwargs = {"activation_fn": torch.nn.ReLU, "net_arch": dict(pi=[128, 128], vf=[128, 128])}

        # Create PPO model
        model = PPO(
            "MlpPolicy",  # Use standard MLP policy for now
            env,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            clip_range=self.clip_range,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=self.device,
            tensorboard_log=str(log_path),
        )

        print(f"✓ PPO model created with {total_timesteps} timesteps")

        # Create evaluation callback
        eval_env = self._create_env()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(model_dir / "evaluations"),
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=5,
            verbose=verbose,
        )

        print("🚀 Starting training...")
        start_time = time.time()

        # Train the model
        model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.1f} seconds")

        # Save final model
        final_model_path = model_dir / "final_model"
        model.save(final_model_path)
        print(f"✓ Final model saved to: {final_model_path}")

        # Save training metadata
        metadata = {
            "model_name": model_name,
            "total_timesteps": total_timesteps,
            "training_time_seconds": training_time,
            "reward_function": self.reward_function.__class__.__name__,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "n_steps": self.n_steps,
                "batch_size": self.batch_size,
                "n_epochs": self.n_epochs,
                "clip_range": self.clip_range,
            },
            "environment": {
                "num_players": self.num_players,
                "games_per_episode": self.games_per_episode,
                "action_space": 1365,
                "observation_space": 168,
            },
        }

        import json

        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Metadata saved to: {model_dir / 'metadata.json'}")

        return model, str(model_dir)


# Convenience function for quick training
def quick_train(
    reward_function: Optional[BaseReward] = None, total_timesteps: int = 10000, model_name: str = "quick_test"
) -> Tuple[PPO, str]:
    """Quick training function for testing.

    Args:
        reward_function: Reward function to use
        total_timesteps: Training timesteps
        model_name: Model name

    Returns:
        Tuple of (model, model_directory)
    """
    if reward_function is None:
        from .rewards import DefaultReward

        reward_function = DefaultReward()

    trainer = Trainer(
        reward_function=reward_function,
        learning_rate=5e-4,  # Faster learning for quick tests
        n_steps=256,  # Smaller rollouts
        games_per_episode=3,  # Fewer games per episode
    )

    return trainer.train(
        total_timesteps=total_timesteps,
        model_name=model_name,
        eval_freq=2000,  # Less frequent evaluation
        verbose=1,
    )
