"""Quick test of training pipeline."""

from bigtwo_rl.training import Trainer
from bigtwo_rl.training.rewards import DefaultReward
from bigtwo_rl.training.hyperparams import DefaultConfig
from bigtwo_rl import standard_observation

if __name__ == "__main__":
    print("Testing training pipeline with short run...")
    trainer = Trainer(
        reward_function=DefaultReward(),
        hyperparams=DefaultConfig(),
        observation_config=standard_observation(),
    )
    model, model_dir = trainer.train(total_timesteps=1000)
    print("✓ Training pipeline works!")
