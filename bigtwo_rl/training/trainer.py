"""Training infrastructure for Big Two PPO agents."""

import os
from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

from ..core.rl_wrapper import BigTwoRLWrapper
from ..core.observation_builder import ObservationConfig
from .hyperparams import BaseConfig
from .rewards import BaseReward
from .callbacks import BigTwoMetricsCallback
from .multi_player_buffer import MultiPlayerExperienceBuffer
from .self_play_callback import SimpleSelfPlayCallback


class ConfigurableBigTwoWrapper(BigTwoRLWrapper):
    """BigTwoRLWrapper with configurable reward function and observations."""

    def __init__(
        self,
        observation_config: ObservationConfig,
        num_players=4,
        games_per_episode=10,
        reward_function=None,
        track_move_history: bool = False,
    ):
        # Pass all configuration directly to parent (now uses true self-play by default)
        super().__init__(
            observation_config,
            reward_function,
            games_per_episode,
            track_move_history,
        )


class Trainer:
    """High-level trainer for Big Two PPO agents."""

    def __init__(
        self,
        reward_function: BaseReward,
        hyperparams: BaseConfig,
        observation_config: ObservationConfig,
        eval_freq: int = 500,
        snapshot_dir: Optional[str] = None,
        snapshot_every_steps: Optional[int] = None,
        enable_bigtwo_metrics: bool = True,
    ):
        """
        Initialize trainer with true self-play training.

        Args:
            reward_function: Reward function instance (e.g., DefaultReward())
            hyperparams: Hyperparameter configuration instance (e.g., DefaultConfig())
            eval_freq: How often to evaluate during training - publishes metrics and saves models
            snapshot_dir: Directory to save model snapshots
            snapshot_every_steps: How often to save snapshots
            observation_config: Custom observation configuration
            enable_bigtwo_metrics: Whether to log Big Two-specific metrics to TensorBoard
        """
        # Store reward function and hyperparameters directly
        self.reward_function = reward_function
        self.reward_name = reward_function.__class__.__name__

        self.config = hyperparams.to_dict()
        self.config_name = hyperparams.__class__.__name__

        self.eval_freq = eval_freq
        self.snapshot_dir = snapshot_dir
        self.snapshot_every_steps = snapshot_every_steps
        self.observation_config = observation_config
        self.enable_bigtwo_metrics = enable_bigtwo_metrics

        # Self-play specific components (always enabled now)
        self.multi_player_buffer = MultiPlayerExperienceBuffer()

    def _make_env(self):
        """Create environment instance with configuration."""
        # Now using unified BigTwoRLWrapper with true self-play enabled by default
        env = BigTwoRLWrapper(
            observation_config=self.observation_config,
            games_per_episode=self.config["games_per_episode"],
            reward_function=self.reward_function,
            track_move_history=False,
        )

        # If maskable PPO is available, wrap env to expose action masks
        try:
            from sb3_contrib.common.wrappers import ActionMasker  # type: ignore

            env = ActionMasker(env, lambda e: e.action_masks())
        except Exception:
            # Fallback: continue without explicit mask wrapper
            pass
        return env

    def train(
        self,
        total_timesteps: int = 50000,
        model_name: Optional[str] = None,
        verbose: bool = True,
    ) -> tuple:
        """
        Train a PPO agent.

        Args:
            total_timesteps: Total training timesteps
            model_name: Name for saved model (auto-generated if None)
            verbose: Whether to print progress

        Returns:
            (model, model_directory): Trained model and save path
        """
        # Setup phase
        model_name = model_name or f"{self.config_name}_{self.reward_name}"
        if verbose:
            self._print_training_info(model_name)

        # Environment setup
        self._setup_opponent_provider()
        env, eval_env = self._setup_training_environments()

        # Model creation
        model = self._create_model_instance(env, model_name, verbose)

        # Callback setup
        models_dir = f"./models/{model_name}"
        os.makedirs(models_dir, exist_ok=True)
        callbacks = self._setup_callbacks(eval_env, models_dir)

        # Training execution
        self._execute_training_loop(
            model, total_timesteps, callbacks, models_dir, verbose
        )

        # Post-training save
        self._save_model_and_metadata(model, models_dir, total_timesteps, verbose)

        return model, models_dir

    def _print_training_info(self, model_name: str) -> None:
        """Print training configuration information."""
        print(
            f"Training with config '{self.config_name}' and reward function '{self.reward_name}'"
        )
        print(f"Config: {self.config}")

    def _setup_opponent_provider(self) -> None:
        """Setup opponent provider if configured."""
        # No opponents needed for true self-play (always enabled now)
        self._opponent_provider = None

    def _setup_training_environments(self) -> tuple:
        """Create training and evaluation environments."""
        # For true self-play, we need to use DummyVecEnv so model reference can be shared
        # SubprocVecEnv runs environments in separate processes, making model sharing difficult
        from stable_baselines3.common.vec_env import DummyVecEnv

        # Create training environment (single process for model sharing)
        env = DummyVecEnv([self._make_env for _ in range(self.config["n_envs"])])

        # Create evaluation environment (match vectorized type with training to avoid warnings)
        eval_env = DummyVecEnv([self._make_env])

        return env, eval_env

    def _create_model_instance(self, env, model_name: str, verbose: bool):
        """Create PPO or MaskablePPO model instance."""
        # Prefer MaskablePPO if available; fallback to standard PPO
        use_maskable = self._check_maskable_ppo_availability()

        model_kwargs = {
            "policy": "MlpPolicy",
            "env": env,
            "learning_rate": self.config["learning_rate"],
            "n_steps": self.config["n_steps"],
            "batch_size": self.config["batch_size"],
            "n_epochs": self.config["n_epochs"],
            "gamma": self.config["gamma"],
            "gae_lambda": self.config["gae_lambda"],
            "clip_range": self.config["clip_range"],
            "verbose": 1 if verbose else 0,
            "tensorboard_log": f"./logs/{model_name}/",
        }

        if use_maskable:
            from sb3_contrib.ppo_mask import MaskablePPO  # type: ignore

            return MaskablePPO(**model_kwargs)
        else:
            return PPO(**model_kwargs)

    def _check_maskable_ppo_availability(self) -> bool:
        """Check if MaskablePPO is available."""
        try:
            from sb3_contrib.ppo_mask import MaskablePPO  # type: ignore

            return True
        except Exception:
            return False

    def _setup_callbacks(self, eval_env, models_dir: str) -> list:
        """Setup training callbacks (evaluation, metrics, snapshots)."""
        # Setup evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=models_dir,
            log_path=f"./logs/{os.path.basename(models_dir)}/",
            eval_freq=self.eval_freq,
            deterministic=True,
            render=False,
        )

        callbacks = [eval_callback]

        # Add Big Two metrics callback if enabled
        if self.enable_bigtwo_metrics:
            callbacks.append(BigTwoMetricsCallback(verbose=0))

        # Add self-play callback to monitor multi-player experience collection
        callbacks.append(SimpleSelfPlayCallback(verbose=1))

        # Add snapshot callback if configured
        if self.snapshot_every_steps is not None and self.snapshot_every_steps > 0:
            snapshot_cb = self._create_snapshot_callback(models_dir)
            callbacks.append(snapshot_cb)

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

    def _execute_training_loop(
        self,
        model,
        total_timesteps: int,
        callbacks: list,
        models_dir: str,
        verbose: bool,
    ) -> None:
        """Execute the main training loop."""
        if verbose:
            print(f"Starting PPO training for {total_timesteps} timesteps...")
            print(f"Model will be saved in: {models_dir}")

        model.learn(
            total_timesteps=total_timesteps, callback=callbacks, progress_bar=False
        )

    def _save_model_and_metadata(
        self, model, models_dir: str, total_timesteps: int, verbose: bool
    ) -> None:
        """Save final model and metadata."""
        # Save final model
        model.save(f"{models_dir}/final_model")

        # Save model metadata including observation config
        if self.observation_config is not None:
            from ..agents.model_metadata import ModelMetadata

            # Get observation config from a test environment
            test_env = self._make_env()
            additional_info = {
                "reward_function": self.reward_name,
                "hyperparams": self.config_name,
                "total_timesteps": total_timesteps,
            }
            ModelMetadata.save_metadata(
                models_dir, test_env.obs_config, additional_info
            )

        if verbose:
            print(f"Training completed! Model saved in {models_dir}")
