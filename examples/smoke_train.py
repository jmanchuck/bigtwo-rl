from bigtwo_rl.training import Trainer
from bigtwo_rl.training.rewards import ScoreMarginReward
from bigtwo_rl.training.hyperparams import DefaultConfig
from bigtwo_rl import standard_observation


def main():
    trainer = Trainer(
        reward_function=ScoreMarginReward(),
        hyperparams=DefaultConfig(),
        observation_config=standard_observation(),
        controlled_player=0,
        opponent_mixture={"snapshots": 0.0, "greedy": 0.5, "random": 0.5},
        snapshot_dir="./models/smoke",
        snapshot_every_steps=1000,
        eval_freq=1000,
    )

    model, model_dir = trainer.train(
        total_timesteps=2000, model_name="smoke_test", verbose=True
    )
    print("Model saved to:", model_dir)


if __name__ == "__main__":
    main()
