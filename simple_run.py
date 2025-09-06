from bigtwo_rl.core.observation_builder import strategic_observation
from bigtwo_rl.evaluation.evaluator import Evaluator
from bigtwo_rl.training import Trainer
from bigtwo_rl.training.hyperparams import ReferenceExactConfig
from bigtwo_rl.training.rewards import ZeroSumReward

if __name__ == "__main__":
    # Override config to use single environment to avoid multiprocessing issues
    config = ReferenceExactConfig()
    config.n_envs = 8

    # Train model with true self-play (single environment)
    trainer = Trainer(
        reward_function=ZeroSumReward(),
        hyperparams=config,
        observation_config=strategic_observation(),
    )

    model, model_dir = trainer.train(total_timesteps=1000)

    # Evaluate trained model
    evaluator = Evaluator(num_games=500, n_processes=8)
    results = evaluator.evaluate_model(f"{model_dir}/best_model.zip")

    print(f"Win rate: {results['win_rates']}")
    print(f"Average cards left: {results['avg_cards_left']}")
