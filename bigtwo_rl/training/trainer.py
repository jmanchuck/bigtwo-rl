"""Big Two Trainer with 1,365-action space.

This trainer provides proper action masking and enhanced training capabilities
using the fixed 1,365-action space for optimal RL performance.
"""

import os
import time
from typing import Any

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from bigtwo_rl.core.bigtwo_wrapper import BigTwoWrapper
from bigtwo_rl.core.observation_builder import ObservationConfig
from bigtwo_rl.training.hyperparams import BaseConfig
from bigtwo_rl.training.rewards.base_reward import BaseReward

from .callbacks import BigTwoMetricsCallback
from .masked_policy import MaskedBigTwoPolicy
from .multi_player_ppo import MultiPlayerPPO
from .self_play_callback import SelfPlayPPOCallback


class Trainer:
    """Big Two trainer using 1,365-action space.

    This trainer provides proper action masking and enhanced training capabilities
    using the fixed 1,365-action space for optimal RL performance.
    """

    def __init__(
        self,
        reward_function: BaseReward,
        hyperparams: BaseConfig,
        observation_config: ObservationConfig,
        **kwargs,
    ) -> None:
        """Initialize Fixed Action Trainer.

        Args:
            reward_function: Custom reward function
            hyperparams: Hyperparameter configuration
            observation_config: Observation configuration
            **kwargs: Additional arguments

        """
        # Store core config
        self.reward_function = reward_function
        self.reward_name = reward_function.__class__.__name__
        self.hyperparams = hyperparams
        self.config = hyperparams.to_dict()
        self.config_name = hyperparams.__class__.__name__

        self.observation_config = observation_config

        # Enhanced settings
        self.verbose = kwargs.get("verbose", 1)
        self.model_save_dir = kwargs.get("model_save_dir", "./models")
        self.tensorboard_log_dir = kwargs.get("tensorboard_log_dir", "./logs")

    def _create_model_instance(self, env, model_name: str, verbose: bool) -> MultiPlayerPPO:
        """Create PPO model instance with masked policy.

        Args:
            env: Training environment
            model_name: Name for model logging
            verbose: Verbosity level

        Returns:
            MultiPlayerPPO model with masked policy

        """
        tb_log = os.path.join(self.tensorboard_log_dir, model_name)

        # Validate action space
        if env.action_space.n != 1365:
            raise ValueError(f"Expected 1365 actions, got {env.action_space.n}")

        return MultiPlayerPPO(
            policy=MaskedBigTwoPolicy,  # Our new policy with masking
            env=env,  # BigTwoWrapper
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

    def _make_env(self) -> ActionMasker:
        """Create environment instance with fixed action space.

        Returns:
            Environment with fixed 1365-action space

        """
        env = BigTwoWrapper(
            observation_config=self.observation_config,
            games_per_episode=self.config["games_per_episode"],
            reward_function=self.reward_function,
            track_move_history=False,
        )
        return ActionMasker(env, lambda e: e.get_action_mask())

    def train(self, total_timesteps: int = 50000, **kwargs) -> tuple[MultiPlayerPPO, str]:
        """Train with fixed action space.

        Args:
            total_timesteps: Total training timesteps
            **kwargs: Additional training arguments

        Returns:
            Tuple of (trained_model, model_directory)

        """
        print("🎯 Training with fixed 1,365-action space")
        print(f"📊 Action space size: {self._make_env().action_space.n}")

        # Validate environment
        test_env = self._make_env()
        if test_env.action_space.n != 1365:
            raise ValueError(f"Environment has wrong action space size: {test_env.action_space.n}")

        # Build environments
        env, eval_env = self._setup_training_environments()

        # Create model
        model_name = f"{self.config_name}_{self.reward_name}"
        model = self._create_model_instance(env, model_name, verbose=self.verbose >= 1)

        # Prepare callbacks
        models_dir = os.path.join(self.model_save_dir, model_name)
        os.makedirs(models_dir, exist_ok=True)
        callbacks = self._setup_callbacks(eval_env, models_dir)

        # Learn
        start_time = time.time()
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=kwargs.get("progress_bar", True))
        end_time = time.time()

        # Save
        self._save_model_and_metadata(model, models_dir, total_timesteps)

        # Additional fixed action space validation
        self._validate_training(model, models_dir)

        training_time = end_time - start_time
        print(f"✅ Training completed in {training_time:.1f}s")
        print(f"📁 Model saved to: {models_dir}")

        return model, models_dir

    def _setup_training_environments(self):
        """Create training and evaluation environments."""

        def make_env(is_eval=False):
            env = self._make_env()
            if is_eval:
                env = Monitor(env)
            return env

        env = DummyVecEnv([lambda: make_env() for _ in range(self.config["n_envs"])])
        eval_env = DummyVecEnv([lambda: make_env(is_eval=True)])
        return env, eval_env

    def _setup_callbacks(self, eval_env, models_dir: str):
        """Setup training callbacks (evaluation, metrics, snapshots)."""
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=models_dir,
            log_path=os.path.join(self.tensorboard_log_dir, os.path.basename(models_dir)),
            eval_freq=500,  # Default eval frequency
            deterministic=True,
            render=False,
            verbose=0,
        )
        callbacks = [eval_callback]
        callbacks.append(BigTwoMetricsCallback(verbose=0))
        callbacks.append(SelfPlayPPOCallback(verbose=1))
        return callbacks

    def _save_model_and_metadata(self, model, models_dir: str, total_timesteps: int):
        """Save final model and metadata."""
        model.save(f"{models_dir}/final_model")

        if self.observation_config is not None:
            from ..agents.model_metadata import ModelMetadata

            test_env = self._make_env()
            # Handle wrapped environments (e.g., ActionMasker)
            env_to_check = test_env
            while hasattr(env_to_check, "env") and not hasattr(env_to_check, "obs_config"):
                env_to_check = env_to_check.env

            additional_info = {
                "reward_function": self.reward_name,
                "hyperparams": self.config_name,
                "total_timesteps": total_timesteps,
            }
            ModelMetadata.save_metadata(models_dir, env_to_check.obs_config, additional_info)

    def _validate_training(self, model: MultiPlayerPPO, model_dir: str):
        """Validate that training worked correctly.

        Args:
            model: Trained model
            model_dir: Model directory

        """
        print("🔍 Validating training...")

        # Check model action space
        if model.action_space.n != 1365:
            raise ValueError(f"Model has wrong action space: {model.action_space.n}")

        # Check policy type
        if not isinstance(model.policy, MaskedBigTwoPolicy):
            print(f"⚠️  Warning: Model policy is {type(model.policy)}, expected MaskedBigTwoPolicy")

        # Test action masking
        test_env = self._make_env()
        obs, _ = test_env.reset()
        action_mask = test_env.action_masks()

        # Verify mask properties
        valid_actions = action_mask.sum()
        if valid_actions == 0:
            print("⚠️  Warning: No valid actions in initial state")
        elif valid_actions > 100:
            print(f"ℹ️  Initial state has {valid_actions} valid actions")

        # Test model prediction
        try:
            action, _ = model.predict(obs, deterministic=True)
            if not action_mask[action]:
                print(f"⚠️  Warning: Model predicted invalid action {action}")
            else:
                print("✅ Model prediction validation passed")
        except Exception as e:
            print(f"⚠️  Warning: Model prediction failed: {e}")

        # Check if model files exist
        expected_files = ["final_model.zip", "best_model.zip"]
        for filename in expected_files:
            filepath = os.path.join(model_dir, filename)
            if os.path.exists(filepath):
                print(f"✅ Found {filename}")
            else:
                print(f"⚠️  Missing {filename}")

    def get_training_info(self) -> dict[str, Any]:
        """Get information about this trainer's configuration.

        Returns:
            Dictionary with trainer information

        """
        info = {
            "trainer_type": "Trainer",
            "action_space_type": "fixed",
            "action_space_size": 1365,
            "uses_action_masking": True,
            "policy_type": "MaskedBigTwoPolicy",
            "supports_invalid_actions": False,  # All actions are properly masked
        }

        return info


class EnhancedTrainer(Trainer):
    """Enhanced version with additional training features.

    This trainer adds extra functionality like advanced logging,
    action distribution tracking, and training diagnostics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.track_action_distributions = kwargs.get("track_action_distributions", False)
        self.action_stats = {}

    def train(self, total_timesteps: int = 50000, **kwargs) -> tuple[MultiPlayerPPO, str]:
        """Enhanced training with additional monitoring.

        Args:
            total_timesteps: Total training timesteps
            **kwargs: Additional arguments

        Returns:
            Tuple of (trained_model, model_directory)

        """
        print("🚀 Starting enhanced fixed action space training")

        if self.track_action_distributions:
            print("📊 Action distribution tracking enabled")
            self._setup_action_tracking()

        # Run base training
        model, model_dir = super().train(total_timesteps, **kwargs)

        # Additional analysis
        if self.track_action_distributions:
            self._analyze_action_distributions(model_dir)

        return model, model_dir

    def _setup_action_tracking(self):
        """Setup action distribution tracking."""
        self.action_stats = {
            "action_counts": {},
            "hand_type_preferences": {},
            "mask_statistics": {"total_masks": 0, "avg_valid_actions": 0},
        }

    def _analyze_action_distributions(self, model_dir: str):
        """Analyze action distributions from training.

        Args:
            model_dir: Model directory for saving analysis

        """
        print("📈 Analyzing action distributions...")

        # This would analyze action usage patterns, hand type preferences, etc.
        # For now, just placeholder output
        analysis = {
            "total_actions_analyzed": self.action_stats.get("total_actions", 0),
            "most_used_actions": "Analysis not yet implemented",
            "hand_type_distribution": "Analysis not yet implemented",
            "masking_efficiency": "Analysis not yet implemented",
        }

        # Save analysis
        import json

        analysis_path = os.path.join(model_dir, "action_analysis.json")
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)

        print(f"📊 Action analysis saved to: {analysis_path}")


# Factory function for trainer creation
def create_trainer(use_fixed_actions: bool = True, enhanced: bool = False, **kwargs) -> Trainer:
    """Factory to create trainer based on preferences.

    Args:
        use_fixed_actions: If True, use fixed action space trainer
        enhanced: If True, use enhanced trainer with extra features
        **kwargs: Arguments passed to trainer constructor

    Returns:
        Appropriate trainer instance

    """
    if use_fixed_actions:
        if enhanced:
            return EnhancedTrainer(**kwargs)
        return Trainer(**kwargs)
    # Fall back to legacy trainer
    return Trainer(**kwargs)
