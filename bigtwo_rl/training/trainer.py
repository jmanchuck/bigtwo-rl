"""Training infrastructure for Big Two PPO agents."""

import os
from typing import Optional, Any
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

from ..core.rl_wrapper import BigTwoRLWrapper
from .hyperparams import BaseConfig
from .rewards import BaseReward
from .opponent_pool import OpponentPool, EnvOpponentProvider


class ConfigurableBigTwoWrapper(BigTwoRLWrapper):
    """BigTwoRLWrapper with configurable reward function and observations."""

    def __init__(
        self,
        num_players=4,
        games_per_episode=10,
        reward_function=None,
        controlled_player: int = 0,
        opponent_provider=None,
        observation_config: Optional[Any] = None,
    ):
        # Pass all configuration directly to parent
        super().__init__(
            num_players,
            games_per_episode,
            reward_function,
            controlled_player,
            opponent_provider,
            observation_config,
        )


class Trainer:
    """High-level trainer for Big Two PPO agents."""

    def __init__(
        self,
        reward_function: BaseReward,
        hyperparams: BaseConfig,
        eval_freq: int = 5000,
        controlled_player: int = 0,
        opponent_mixture: Optional[dict] = None,
        snapshot_dir: Optional[str] = None,
        snapshot_every_steps: Optional[int] = None,
        observation_config: Optional[Any] = None,
    ):
        """
        Initialize trainer.

        Args:
            reward_function: Reward function instance (e.g., DefaultReward())
            hyperparams: Hyperparameter configuration instance (e.g., DefaultConfig())
            eval_freq: How often to evaluate during training
            controlled_player: Which player the agent controls (0-3)
            opponent_mixture: Dict specifying opponent mix ratios
            snapshot_dir: Directory to save model snapshots
            snapshot_every_steps: How often to save snapshots
            observation_config: Custom observation configuration
        """
        # Store reward function and hyperparameters directly
        self.reward_function = reward_function
        self.reward_name = reward_function.__class__.__name__

        self.config = hyperparams.to_dict()
        self.config_name = hyperparams.__class__.__name__

        self.eval_freq = eval_freq
        self.controlled_player = controlled_player
        self.opponent_mixture = opponent_mixture
        self.snapshot_dir = snapshot_dir
        self.snapshot_every_steps = snapshot_every_steps
        self.observation_config = observation_config
        self._opponent_provider = None

    def _make_env(self):
        """Create environment instance with configuration."""
        return ConfigurableBigTwoWrapper(
            games_per_episode=self.config["games_per_episode"],
            reward_function=self.reward_function,
            controlled_player=self.controlled_player,
            opponent_provider=self._opponent_provider,
            observation_config=self.observation_config,
        )

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

        # Create model name if not provided
        if model_name is None:
            model_name = f"{self.config_name}_{self.reward_name}"

        if verbose:
            print(
                f"Training with config '{self.config_name}' and reward function '{self.reward_name}'"
            )
            print(f"Config: {self.config}")

        # Prepare opponent provider (optional)
        if self.snapshot_dir is not None or self.opponent_mixture is not None:
            snapshot_dir = self.snapshot_dir or "./models"
            pool = OpponentPool(
                snapshot_dir=snapshot_dir, mixture=self.opponent_mixture
            )
            self._opponent_provider = EnvOpponentProvider(pool)
        else:
            self._opponent_provider = None

        # Create training environment with true multiprocessing
        env = SubprocVecEnv([self._make_env for _ in range(self.config["n_envs"])])

        # Create evaluation environment
        eval_env = make_vec_env(self._make_env, n_envs=1)

        # PPO configuration from config
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.config["learning_rate"],
            n_steps=self.config["n_steps"],
            batch_size=self.config["batch_size"],
            n_epochs=self.config["n_epochs"],
            gamma=self.config["gamma"],
            gae_lambda=self.config["gae_lambda"],
            clip_range=self.config["clip_range"],
            verbose=1 if verbose else 0,
            tensorboard_log=f"./logs/{model_name}/",
        )

        # Setup evaluation and optional snapshotting callbacks
        models_dir = f"./models/{model_name}"
        os.makedirs(models_dir, exist_ok=True)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=models_dir,
            log_path=f"./logs/{model_name}/",
            eval_freq=self.eval_freq,
            deterministic=True,
            render=False,
        )

        callback = eval_callback
        if self.snapshot_every_steps is not None and self.snapshot_every_steps > 0:

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
            snapshot_cb = SnapshotCallback(
                save_dir=snapshot_dir, freq=self.snapshot_every_steps
            )
            callback = [eval_callback, snapshot_cb]

        if verbose:
            print(f"Starting PPO training for {total_timesteps} timesteps...")
            print(f"Model will be saved in: {models_dir}")

        model.learn(
            total_timesteps=total_timesteps, callback=callback, progress_bar=False
        )

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

        return model, models_dir
