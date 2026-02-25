"""Training pipeline for Big Two agents with 1,365-action space."""

import time
from pathlib import Path
from typing import Any

import torch
from stable_baselines3.common.callbacks import CallbackList

from bigtwo_rl.training.rewards import DefaultReward
from bigtwo_rl.training.rewards.base_reward import BaseReward
from bigtwo_rl.training.multi_player_ppo import MultiPlayerPPO
from bigtwo_rl.training.self_play_callback import SimpleSelfPlayCallback

# Try to import stable-baselines3 components
try:
    from stable_baselines3.common.callbacks import EvalCallback

    SB3_AVAILABLE = True
except ImportError:
    EvalCallback: Any | None = None
    SB3_AVAILABLE = False


class Trainer:
    """Training pipeline for Big Two agents."""

    def __init__(
        self,
        reward_function: BaseReward | None = None,
        num_players: int = 4,
        games_per_episode: int = 5,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        n_steps: int = 512,
        batch_size: int = 64,
        n_epochs: int = 10,
        clip_range: float = 0.2,
        device: str = "auto",
        observation_mode: str = "enhanced",
        policy_net_arch: dict[str, list[int]] | None = None,
        n_envs: int = 1,
        ent_coef: float = 0.005,
        league_opponent_prob: float = 0.0,
        snapshot_interval_rollouts: int = 20,
        snapshot_max_policies: int = 8,
        anneal_lr: bool = True,
        anneal_clip: bool = True,
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
        if observation_mode != "enhanced":
            raise ValueError("Only observation_mode='enhanced' is supported")
        self.observation_mode = observation_mode
        self.policy_net_arch = policy_net_arch or {"pi": [128, 128], "vf": [128, 128]}
        self.n_envs = n_envs
        self.ent_coef = ent_coef
        self.league_opponent_prob = league_opponent_prob
        self.snapshot_interval_rollouts = snapshot_interval_rollouts
        self.snapshot_max_policies = snapshot_max_policies
        self.anneal_lr = anneal_lr
        self.anneal_clip = anneal_clip

        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"✓ Trainer initialized with device: {self.device}")

    def _create_env(self):
        """Create training environment."""
        # Import here to avoid circular import
        from ..core.bigtwo_wrapper import BigTwoWrapper

        return BigTwoWrapper(
            reward_function=self.reward_function,
            num_players=self.num_players,
            games_per_episode=self.games_per_episode,
            track_move_history=True,
            observation_mode=self.observation_mode,
        )

    def _make_env_fn(self):
        """Create environment factory for vectorized envs."""

        def _fn():
            return self._create_env()

        return _fn

    def train(
        self,
        total_timesteps: int = 25000,
        model_name: str | None = None,
        log_dir: str = "./logs",
        save_dir: str = "./models",
        eval_freq: int = 5000,
        verbose: int = 1,
    ) -> tuple[Any, str]:
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

        # Create vectorized environment (reference-style parallel games)
        from stable_baselines3.common.vec_env import DummyVecEnv

        env = DummyVecEnv([self._make_env_fn() for _ in range(max(1, self.n_envs))])

        # Create policy with action masking support
        policy_kwargs = {"activation_fn": torch.nn.ReLU, "net_arch": self.policy_net_arch}

        lr = self.learning_rate
        if self.anneal_lr:
            lr_init = self.learning_rate
            lr = lambda progress: max(1e-6, lr_init * progress)

        clip = self.clip_range
        if self.anneal_clip:
            clip_init = self.clip_range
            clip = lambda progress: clip_init * progress

        # Create PPO model with multi-player rollout handling
        model = MultiPlayerPPO(
            "MlpPolicy",
            env,
            learning_rate=lr,
            gamma=self.gamma,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            clip_range=clip,
            ent_coef=self.ent_coef,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=self.device,
            tensorboard_log=str(log_path),
            league_opponent_prob=self.league_opponent_prob,
            snapshot_interval_rollouts=self.snapshot_interval_rollouts,
            snapshot_max_policies=self.snapshot_max_policies,
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

        self_play_callback = SimpleSelfPlayCallback(verbose=verbose)
        callbacks = CallbackList([self_play_callback, eval_callback])

        # Train the model
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)

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
                "ent_coef": self.ent_coef,
            },
            "environment": {
                "num_players": self.num_players,
                "games_per_episode": self.games_per_episode,
                "action_space": 1365,
                "observation_space": int(env.observation_space.shape[0]),
            },
        }

        import json

        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Metadata saved to: {model_dir / 'metadata.json'}")

        return model, str(model_dir)


# Convenience function for quick training
def quick_train(
    reward_function: BaseReward | None = None,
    total_timesteps: int = 10000,
    model_name: str = "quick_test",
) -> tuple[Any, str]:
    """Quick training function for testing.

    Args:
        reward_function: Reward function to use
        total_timesteps: Training timesteps
        model_name: Model name

    Returns:
        Tuple of (model, model_directory)

    """
    if reward_function is None:
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
