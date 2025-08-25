"""Enhanced trainer interface for multi-player Big Two training.

This module provides a clean API that integrates all multi-player enhancements:
- MultiPlayerPPO with delayed reward assignment
- MultiPlayerRolloutBuffer for proper buffer management
- MultiPlayerGAECallback for turn-based GAE calculation
- Reference-compatible configurations

The API maintains compatibility with the standard Trainer while providing
significant algorithmic improvements for turn-based games.
"""

import os
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ..core.observation_builder import ObservationConfig
from ..core.rl_wrapper import BigTwoRLWrapper
from .callbacks import BigTwoMetricsCallback
from .hyperparams import BaseConfig
from .multi_player_ppo import MultiPlayerPPO
from .rewards import BaseReward
from .self_play_callback import SelfPlayPPOCallback


class Trainer:
    """Enhanced Big Two trainer with multi-player algorithmic improvements.

    This is the main Trainer class that provides:
    - Multi-player PPO with delayed reward assignment
    - Turn-based GAE calculation for 4-player games
    - Reference implementation compatibility
    - Enhanced statistics and logging

    The trainer uses multi-player enhancements by default but can fall back
    to standard algorithms if needed. All existing APIs remain the same.

    Key Features:
    - Multi-player aware algorithms (enabled by default)
    - Delayed reward assignment matching reference implementation
    - 4-player GAE calculation for turn-based games
    - Clean separation of concerns with modular design
    - Comprehensive statistics and monitoring
    """

    def __init__(
        self,
        reward_function: BaseReward,
        hyperparams: BaseConfig,
        observation_config: ObservationConfig,
        eval_freq: int = 500,
        snapshot_dir: str | None = None,
        snapshot_every_steps: int | None = None,
        enable_bigtwo_metrics: bool = True,
        verbose: int = 1,
        model_save_dir: str = "./models",
        tensorboard_log_dir: str = "./logs",
    ):
        """Initialize enhanced Trainer with multi-player capabilities.

        Args:
            reward_function: Reward function for training
            hyperparams: Training hyperparameters
            observation_config: Observation space configuration
            eval_freq: Evaluation frequency during training
            snapshot_dir: Directory to save model snapshots
            snapshot_every_steps: Save snapshot every N steps
            enable_bigtwo_metrics: Enable Big Two specific metrics logging
            verbose: Verbosity level
            model_save_dir: Base directory to save models
            tensorboard_log_dir: Base directory to save logs

        """
        # Store core config
        self.reward_function = reward_function
        self.reward_name = reward_function.__class__.__name__
        self.hyperparams = hyperparams
        self.config = hyperparams.to_dict()
        self.config_name = hyperparams.__class__.__name__

        self.observation_config = observation_config
        self.eval_freq = eval_freq
        self.snapshot_dir = snapshot_dir
        self.snapshot_every_steps = snapshot_every_steps
        self.enable_bigtwo_metrics = enable_bigtwo_metrics

        # Enhanced settings
        self.verbose = verbose
        self.model_save_dir = model_save_dir
        self.tensorboard_log_dir = tensorboard_log_dir

        # Enable multi-player enhancements by default (matching reference implementation)
        self.enable_multi_player_enhancements = True

        # Trainer initialized with multi-player algorithms

    def _create_model_instance(self, env, model_name: str, verbose: bool) -> MultiPlayerPPO:
        """Create PPO model instance.

        Uses MultiPlayerPPO when enhancements are enabled, otherwise falls back
        to standard PPO/MaskablePPO selection.
        """
        tb_log = os.path.join(self.tensorboard_log_dir, model_name)

        model = MultiPlayerPPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.config["learning_rate"],
            n_steps=self.config["n_steps"],
            batch_size=self.config["batch_size"],
            n_epochs=self.config["n_epochs"],
            gamma=self.config["gamma"],
            gae_lambda=self.config["gae_lambda"],
            clip_range=self.config["clip_range"],
            verbose=1 if verbose else 0,
            tensorboard_log=tb_log,
            device="auto",
        )

        # MultiPlayerPPO created with enhanced algorithms
        return model

    def _make_env(self, is_eval=False):
        """Create environment instance with configuration."""
        env = BigTwoRLWrapper(
            observation_config=self.observation_config,
            games_per_episode=self.config["games_per_episode"],
            reward_function=self.reward_function,
            track_move_history=False,
        )

        # Wrap with Monitor for evaluation environments to suppress warnings
        if is_eval:
            env = Monitor(env)

        try:
            from sb3_contrib.common.wrappers import ActionMasker  # type: ignore

            env = ActionMasker(env, lambda e: e.action_masks())
        except Exception:
            pass
        return env

    def _setup_training_environments(self) -> tuple:
        """Create training and evaluation environments."""
        env = DummyVecEnv([self._make_env for _ in range(self.config["n_envs"])])
        eval_env = DummyVecEnv([lambda: self._make_env(is_eval=True)])
        return env, eval_env

    def _setup_callbacks(self, eval_env, models_dir: str) -> list:
        """Setup training callbacks (evaluation, metrics, snapshots)."""
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=models_dir,
            log_path=os.path.join(self.tensorboard_log_dir, os.path.basename(models_dir)),
            eval_freq=self.eval_freq,
            deterministic=True,
            render=False,
            verbose=0,  # Suppress evaluation logs
        )
        callbacks = [eval_callback]
        if self.enable_bigtwo_metrics:
            callbacks.append(BigTwoMetricsCallback(verbose=0))
        callbacks.append(SelfPlayPPOCallback(verbose=1))

        if self.snapshot_every_steps is not None and self.snapshot_every_steps > 0:
            callbacks.append(self._create_snapshot_callback(models_dir))

        return callbacks

    def _create_snapshot_callback(self, models_dir: str):
        """Create snapshot callback for periodic model saving."""

        class SnapshotCallback(BaseCallback):
            def __init__(self, save_dir: str, freq: int, verbose: int = 0):
                super().__init__(verbose)
                self.save_dir = save_dir
                self.freq = freq
                os.makedirs(self.save_dir, exist_ok=True)

            def _on_step(self) -> bool:
                if self.n_calls % self.freq == 0:
                    path = os.path.join(self.save_dir, f"step_{self.n_calls}")
                    os.makedirs(path, exist_ok=True)
                    self.model.save(os.path.join(path, "model"))
                return True

        snapshot_dir = self.snapshot_dir or models_dir
        return SnapshotCallback(save_dir=snapshot_dir, freq=self.snapshot_every_steps)

    def train(
        self,
        total_timesteps: int = 50000,
        callback=None,
        progress_bar: bool = True,
    ) -> tuple[MultiPlayerPPO, str]:
        """Train the model.

        Maintains the legacy public interface while using multi-player enhancements
        by default.
        """
        # Training started

        # Build environments
        env, eval_env = self._setup_training_environments()

        # Create model
        model_name = f"{self.config_name}_{self.reward_name}"
        model = self._create_model_instance(env, model_name, verbose=self.verbose >= 1)

        # Prepare callbacks
        models_dir = os.path.join(self.model_save_dir, model_name)
        os.makedirs(models_dir, exist_ok=True)
        callbacks = self._setup_callbacks(eval_env, models_dir)
        if callback is not None:
            callbacks = callbacks + ([callback] if not isinstance(callback, list) else callback)

        # Learn
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=progress_bar)

        # Save
        self._save_model_and_metadata(model, models_dir, total_timesteps)

        # Post-train stats
        if self.enable_multi_player_enhancements and hasattr(model, "get_multi_player_statistics"):
            stats = model.get_multi_player_statistics()
            # Enhanced training statistics available in stats

        # Expose model on trainer
        self.model = model
        return model, models_dir

    def get_training_statistics(self) -> dict[str, Any]:
        """Get comprehensive training statistics.

        Returns enhanced statistics including multi-player metrics when available.
        """
        stats = {}

        # Add multi-player specific statistics if available
        if (
            self.enable_multi_player_enhancements
            and hasattr(self, "model")
            and hasattr(self.model, "get_multi_player_statistics")
        ):
            mp_stats = self.model.get_multi_player_statistics()
            stats.update({f"enhanced_{k}": v for k, v in mp_stats.items()})

        # Add configuration information
        stats.update(
            {
                "multi_player_enhancements_enabled": self.enable_multi_player_enhancements,
                "trainer_type": "Trainer",
                "ppo_type": "MultiPlayerPPO" if self.enable_multi_player_enhancements else "StandardPPO",
            },
        )

        return stats

    @classmethod
    def create_reference_compatible(
        cls,
        reward_function: BaseReward,
        observation_config: ObservationConfig,
        model_save_dir: str = "./models",
        tensorboard_log_dir: str = "./logs",
        verbose: int = 1,
    ) -> "Trainer":
        """Create a trainer with reference-compatible settings.

        This factory method creates a MultiPlayerTrainer configured to match
        the reference implementation as closely as possible.

        Args:
            reward_function: Reward function (should be zero-sum for best results)
            observation_config: Observation configuration
            model_save_dir: Directory to save models
            tensorboard_log_dir: Directory for logs
            verbose: Verbosity level

        Returns:
            MultiPlayerTrainer configured for reference compatibility

        """
        # Import here to avoid circular imports
        from .hyperparams import ReferenceExactConfig

        # Create reference-matched hyperparameters
        hyperparams = ReferenceExactConfig()

        # Create trainer with enhancements enabled
        trainer = cls(
            reward_function=reward_function,
            hyperparams=hyperparams,
            observation_config=observation_config,
            enable_bigtwo_metrics=True,
            model_save_dir=model_save_dir,
            tensorboard_log_dir=tensorboard_log_dir,
            verbose=verbose,
        )

        # Reference-compatible trainer created

        return trainer

    def __repr__(self) -> str:
        """String representation of the trainer."""
        enhancements = "ENABLED" if self.enable_multi_player_enhancements else "DISABLED"
        return (
            f"Trainer(enhancements={enhancements}, "
            f"reward={type(self.reward_function).__name__}, "
            f"hyperparams={type(self.hyperparams).__name__})"
        )

    def save_training_config(self, filepath: str | Path | None = None) -> str:
        """Save complete training configuration for reproducibility.

        Args:
            filepath: Optional path to save config. If None, saves to model directory.

        Returns:
            Path to saved configuration file

        """
        import json
        from datetime import datetime

        if filepath is None:
            filepath = os.path.join("./models", "training_config.json")

        config = {
            "trainer_type": "Trainer",
            "multi_player_enhancements": self.enable_multi_player_enhancements,
            "timestamp": datetime.now().isoformat(),
            "reward_function": {
                "class": type(self.reward_function).__name__,
                "module": type(self.reward_function).__module__,
            },
            "hyperparams": {
                "config": self.config,
                "config_name": getattr(self, "config_name", "Unknown"),
            },
            "observation_config": {
                "total_size": getattr(self.observation_config, "_total_size", "unknown"),
                "features": getattr(self.observation_config, "__dict__", {}),
            },
        }

        # Add enhanced statistics if available
        try:
            enhanced_stats = self.get_training_statistics()
            config["training_statistics"] = enhanced_stats
        except Exception as e:
            config["training_statistics_error"] = str(e)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(config, f, indent=2, default=str)

        # Training configuration saved

        return str(filepath)

    def _save_model_and_metadata(self, model, models_dir: str, total_timesteps: int) -> None:
        """Save final model and metadata."""
        model.save(f"{models_dir}/final_model")

        if self.observation_config is not None:
            from ..agents.model_metadata import ModelMetadata

            test_env = self._make_env()
            additional_info = {
                "reward_function": self.reward_name,
                "hyperparams": self.config_name,
                "total_timesteps": total_timesteps,
            }
            ModelMetadata.save_metadata(models_dir, test_env.obs_config, additional_info)

        # Training completed, model saved
