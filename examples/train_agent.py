#!/usr/bin/env python3
"""Simple example: Train a Big Two PPO agent."""

from bigtwo_rl.training import Trainer

def main():
    """Train an agent with default settings."""
    print("Training a Big Two agent with default settings...")
    
    # Create trainer with default hyperparameters and reward function
    trainer = Trainer(
        reward_function="default",
        hyperparams="default", 
        eval_freq=5000
    )
    
    # Train for 25,000 timesteps
    model, model_dir = trainer.train(
        total_timesteps=25000,
        model_name="example_agent"
    )
    
    print(f"\nTraining complete!")
    print(f"Model saved in: {model_dir}")
    print(f"To evaluate: python examples/evaluate_agent.py {model_dir}/best_model")
    print(f"To run tournament: python examples/tournament_example.py")

if __name__ == "__main__":
    main()