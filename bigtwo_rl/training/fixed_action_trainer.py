"""Fixed Action Trainer for Big Two with 1,365-action space.

This trainer extends the base Trainer class to work with the fixed action space,
providing proper action masking and enhanced training capabilities.
"""

import os
import time
from typing import Any

from sb3_contrib.common.wrappers import ActionMasker

from bigtwo_rl.training.hyperparams import BaseConfig
from bigtwo_rl.training.rewards.base_reward import BaseReward

from ..core.fixed_action_wrapper import FixedActionBigTwoWrapper
from ..core.observation_builder import ObservationConfig
from .masked_policy import MaskedBigTwoPolicy
from .multi_player_ppo import MultiPlayerPPO
from .trainer import Trainer


class FixedActionTrainer(Trainer):
    """Big Two trainer using fixed 1,365-action space.

    This trainer provides the same interface as the base Trainer but uses
    the fixed action space with proper action masking for better learning.
    """

    def __init__(
        self,
        reward_function: BaseReward,
        hyperparams: BaseConfig,
        observation_config: ObservationConfig,
        **kwargs,
    ):
        """Initialize Fixed Action Trainer.

        Args:
            reward_function: Custom reward function
            hyperparams: Hyperparameter configuration
            observation_config: Observation configuration
            **kwargs: Additional arguments

        """
        super().__init__(
            reward_function=reward_function,
            hyperparams=hyperparams,
            observation_config=observation_config,
            **kwargs,
        )

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

        model = MultiPlayerPPO(
            policy=MaskedBigTwoPolicy,  # Our new policy with masking
            env=env,  # FixedActionBigTwoWrapper
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

        return model

    def _make_env(self):
        """Create environment instance with fixed action space.

        Returns:
            Environment with fixed 1365-action space

        """
        env = FixedActionBigTwoWrapper(  # New wrapper!
            observation_config=self.observation_config,
            games_per_episode=self.config["games_per_episode"],
            reward_function=self.reward_function,
            track_move_history=False,
        )

        # Use sb3-contrib ActionMasker for automatic masking
        try:
            env = ActionMasker(env, lambda e: e.get_action_mask())
        except ImportError:
            # Fallback if sb3-contrib not available
            print("Warning: sb3-contrib not available, action masking may not work optimally")
            print("Consider installing: pip install sb3-contrib")

        return env

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

        # Run base training
        start_time = time.time()
        model, model_dir = super().train(total_timesteps, **kwargs)
        end_time = time.time()

        # Additional fixed action space validation
        self._validate_fixed_action_training(model, model_dir)

        training_time = end_time - start_time
        print(f"✅ Fixed action space training completed in {training_time:.1f}s")
        print(f"📁 Model saved to: {model_dir}")

        return model, model_dir

    def _validate_fixed_action_training(self, model: MultiPlayerPPO, model_dir: str):
        """Validate that fixed action space training worked correctly.

        Args:
            model: Trained model
            model_dir: Model directory

        """
        print("🔍 Validating fixed action space training...")

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
        base_info = super().get_training_info() if hasattr(super(), "get_training_info") else {}

        fixed_action_info = {
            "trainer_type": "FixedActionTrainer",
            "action_space_type": "fixed",
            "action_space_size": 1365,
            "uses_action_masking": True,
            "policy_type": "MaskedBigTwoPolicy",
            "supports_invalid_actions": False,  # All actions are properly masked
        }

        # Merge with base info
        base_info.update(fixed_action_info)
        return base_info


class EnhancedFixedActionTrainer(FixedActionTrainer):
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
            return EnhancedFixedActionTrainer(**kwargs)
        return FixedActionTrainer(**kwargs)
    # Fall back to legacy trainer
    return Trainer(**kwargs)


# Migration helper
def migrate_from_legacy_trainer(
    legacy_trainer: Trainer,
    preserve_settings: bool = True,
) -> FixedActionTrainer:
    """Migrate settings from legacy trainer to fixed action trainer.

    Args:
        legacy_trainer: Existing legacy trainer
        preserve_settings: Whether to preserve all settings

    Returns:
        New FixedActionTrainer with migrated settings

    """
    # Extract settings from legacy trainer
    kwargs = {}

    if hasattr(legacy_trainer, "reward_function"):
        kwargs["reward_function"] = legacy_trainer.reward_function
    if hasattr(legacy_trainer, "hyperparams"):
        kwargs["hyperparams"] = legacy_trainer.hyperparams
    if hasattr(legacy_trainer, "observation_config"):
        kwargs["observation_config"] = legacy_trainer.observation_config

    if preserve_settings:
        # Copy additional attributes
        for attr in ["tensorboard_log_dir", "models_dir"]:
            if hasattr(legacy_trainer, attr):
                kwargs[attr] = getattr(legacy_trainer, attr)

    return FixedActionTrainer(**kwargs)
