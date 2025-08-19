#!/usr/bin/env python3
"""Configurable training script for Big Two PPO agents."""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from rl_wrapper import BigTwoRLWrapper
from configs.hyperparams import get_config, list_configs
from configs.rewards import get_reward_function, list_reward_functions


class ConfigurableBigTwoWrapper(BigTwoRLWrapper):
    """BigTwoRLWrapper with configurable reward function."""

    def __init__(self, num_players=4, games_per_episode=10, reward_function_name="default"):
        super().__init__(num_players, games_per_episode)
        self.reward_function = get_reward_function(reward_function_name)
        self.reward_function_name = reward_function_name

    def step(self, action):
        """Override step to use custom reward function."""
        # Get current player before step (for reward calculation)
        player_before_step = self.env.current_player

        # Call parent step method but ignore the rewards
        obs, _, done, truncated, info = super().step(action)

        # Calculate custom rewards if game is done
        if done and hasattr(self, "cumulative_reward"):
            # Game finished - recalculate rewards using custom function
            if hasattr(self.env, "done") and self.env.done:
                # Find winner (player with 0 cards)
                winner_player = None
                all_cards_left = []
                for p in range(self.env.num_players):
                    cards_left = len(self.env.hands[p])
                    all_cards_left.append(cards_left)
                    if cards_left == 0:
                        winner_player = p

                if winner_player is not None:
                    # Calculate reward for the player who just played
                    cards_left = all_cards_left[player_before_step]

                    # Use custom reward function
                    if self.reward_function_name == "ranking":
                        custom_reward = self.reward_function(
                            winner_player, player_before_step, cards_left, all_cards_left
                        )
                    else:
                        custom_reward = self.reward_function(winner_player, player_before_step, cards_left)

                    # Update cumulative reward
                    self.cumulative_reward += custom_reward - self.cumulative_reward  # Replace last reward

                    # If episode is complete, return final reward
                    if self.games_played >= self.games_per_episode:
                        final_reward = self.cumulative_reward / self.games_per_episode
                        return obs, final_reward, True, truncated, info

        # Return the original step result for non-terminal states
        return obs, 0.0, done, truncated, info


def make_env(config, reward_function_name):
    """Create environment instance with configuration."""
    return ConfigurableBigTwoWrapper(
        games_per_episode=config["games_per_episode"], reward_function_name=reward_function_name
    )


def train_agent_with_config(
    config_name="default", reward_function_name="default", total_timesteps=50000, eval_freq=5000, model_name=None
):
    """Train PPO agent with specified configuration and reward function."""

    # Get configuration
    config = get_config(config_name)
    reward_function = get_reward_function(reward_function_name)

    print(f"Training with config '{config_name}' and reward function '{reward_function_name}'")
    print(f"Config: {config}")

    # Create model name if not provided
    if model_name is None:
        model_name = f"{config_name}_{reward_function_name}"

    # Create training environment with true multiprocessing
    env = SubprocVecEnv([lambda: make_env(config, reward_function_name) for _ in range(config["n_envs"])])

    # Create evaluation environment
    eval_env = make_vec_env(lambda: make_env(config, reward_function_name), n_envs=1)

    # PPO configuration from config
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        verbose=1,
        tensorboard_log=f"./logs/{model_name}/",
    )

    # Setup evaluation callback
    models_dir = f"./models/{model_name}"
    os.makedirs(models_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=f"./logs/{model_name}/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )

    print(f"Starting PPO training for {total_timesteps} timesteps...")
    print(f"Model will be saved in: {models_dir}")

    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=False)

    # Save final model
    model.save(f"{models_dir}/final_model")
    print(f"Training completed! Model saved in {models_dir}")

    return model, models_dir


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    config_name = "default"
    reward_function_name = "default"
    timesteps = 50000
    model_name = None

    if len(sys.argv) > 1:
        if sys.argv[1] == "--list-configs":
            print("Available configurations:")
            for config in list_configs():
                print(f"  - {config}")
            print("\nAvailable reward functions:")
            for reward in list_reward_functions():
                print(f"  - {reward}")
            exit(0)
        config_name = sys.argv[1]

    if len(sys.argv) > 2:
        reward_function_name = sys.argv[2]

    if len(sys.argv) > 3:
        timesteps = int(sys.argv[3])

    if len(sys.argv) > 4:
        model_name = sys.argv[4]

    print(f"Training with config '{config_name}', reward '{reward_function_name}', {timesteps} timesteps")

    try:
        model, model_dir = train_agent_with_config(
            config_name=config_name,
            reward_function_name=reward_function_name,
            total_timesteps=timesteps,
            model_name=model_name,
        )
        print(f"\nTraining complete! Model saved in: {model_dir}")
        print(f"To evaluate: python evaluate.py {model_dir}/best_model")
        print(f"To play against: python play_vs_agent.py {model_dir}/best_model")
    except ValueError as e:
        print(f"Error: {e}")
        print(f"\nAvailable configs: {list_configs()}")
        print(f"Available reward functions: {list_reward_functions()}")
