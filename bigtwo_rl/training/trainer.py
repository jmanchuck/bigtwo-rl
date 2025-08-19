"""Training infrastructure for Big Two PPO agents."""

import os
from typing import Union, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from ..core.rl_wrapper import BigTwoRLWrapper
from .hyperparams import get_config, list_configs
from .rewards import BaseReward, get_reward_function, list_reward_functions


class ConfigurableBigTwoWrapper(BigTwoRLWrapper):
    """BigTwoRLWrapper with configurable reward function."""

    def __init__(self, num_players=4, games_per_episode=10, reward_function=None):
        # Pass reward_function directly to parent - it now handles intermediate rewards
        super().__init__(num_players, games_per_episode, reward_function)


class Trainer:
    """High-level trainer for Big Two PPO agents."""
    
    def __init__(self, 
                 reward_function: Union[str, BaseReward, callable] = "default",
                 hyperparams: Union[str, dict] = "default",
                 eval_freq: int = 5000):
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
            self.reward_name = getattr(reward_function, '__name__', 'custom')
            
        # Handle hyperparameters
        if isinstance(hyperparams, str):
            self.config = get_config(hyperparams)
            self.config_name = hyperparams
        else:
            self.config = hyperparams
            self.config_name = "custom"
            
        self.eval_freq = eval_freq
        
    def _make_env(self):
        """Create environment instance with configuration."""
        return ConfigurableBigTwoWrapper(
            games_per_episode=self.config["games_per_episode"], 
            reward_function=self.reward_function
        )
        
    def train(self, 
              total_timesteps: int = 50000,
              model_name: Optional[str] = None,
              verbose: bool = True) -> tuple:
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
            print(f"Training with config '{self.config_name}' and reward function '{self.reward_name}'")
            print(f"Config: {self.config}")

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

        # Setup evaluation callback
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

        if verbose:
            print(f"Starting PPO training for {total_timesteps} timesteps...")
            print(f"Model will be saved in: {models_dir}")

        model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=False)

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