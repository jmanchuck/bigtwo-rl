"""Training infrastructure for Big Two PPO agents."""

import os
from typing import Union, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

from ..core.rl_wrapper import BigTwoRLWrapper
from .hyperparams import get_config, list_configs
from .rewards import BaseReward, get_reward_function, list_reward_functions
from .opponent_pool import OpponentPool, EnvOpponentProvider


class ConfigurableBigTwoWrapper(BigTwoRLWrapper):
    """BigTwoRLWrapper with configurable reward function."""

    def __init__(
        self,
        num_players=4,
        games_per_episode=10,
        reward_function=None,
        controlled_player: int = 0,
        opponent_provider=None,
    ):
        # Pass reward_function directly to parent - it now handles intermediate rewards
        super().__init__(
            num_players,
            games_per_episode,
            reward_function,
            controlled_player,
            opponent_provider,
        )


class Trainer:
    """High-level trainer for Big Two PPO agents."""

    def __init__(
        self,
        reward_function: Union[str, BaseReward, callable] = "default",
        hyperparams: Union[str, dict] = "default",
        eval_freq: int = 5000,
        controlled_player: int = 0,
        opponent_mixture: Optional[dict] = None,
        snapshot_dir: Optional[str] = None,
        snapshot_every_steps: Optional[int] = None,
    ):
        """
        Initialize trainer.

        Args:
            reward_function: Reward function - can be string name, BaseReward instance, or callable
            hyperparams: Hyperparameters - can be string name or dict
            eval_freq: How often to evaluate during training
        """
        # Handle reward function
        if isinstance(reward_function, str):
            self.reward_function = get_reward_function(reward_function)
            self.reward_name = reward_function
        elif isinstance(reward_function, BaseReward):
            self.reward_function = reward_function
            self.reward_name = reward_function.__class__.__name__
        else:
            self.reward_function = reward_function
            self.reward_name = getattr(reward_function, "__name__", "custom")

        # Handle hyperparameters
        if isinstance(hyperparams, str):
            self.config = get_config(hyperparams)
            self.config_name = hyperparams
        else:
            self.config = hyperparams
            self.config_name = "custom"

        self.eval_freq = eval_freq
        self.controlled_player = controlled_player
        self.opponent_mixture = opponent_mixture
        self.snapshot_dir = snapshot_dir
        self.snapshot_every_steps = snapshot_every_steps
        self._opponent_provider = None

    def _make_env(self):
        """Create environment instance with configuration."""
        return ConfigurableBigTwoWrapper(
            games_per_episode=self.config["games_per_episode"],
            reward_function=self.reward_function,
            controlled_player=self.controlled_player,
            opponent_provider=self._opponent_provider,
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

        if verbose:
            print(f"Training completed! Model saved in {models_dir}")

        return model, models_dir

    @staticmethod
    def list_configs():
        """List available hyperparameter configurations."""
        return list_configs()

    @staticmethod
    def list_reward_functions():
        """List available reward functions."""
        return list_reward_functions()
