#!/usr/bin/env python3
"""Example: Train an agent with a custom reward function."""

from bigtwo_rl.training import Trainer, BaseReward

class CustomReward(BaseReward):
    """Example custom reward function that heavily penalizes losing with many cards."""
    
    def calculate(self, winner_player, player_idx, cards_left, all_cards_left=None):
        """Calculate reward with heavy penalties for many remaining cards."""
        if player_idx == winner_player:
            return 10  # Big win bonus
        else:
            # Exponential penalty for cards remaining
            return -(cards_left ** 2) * 0.5

def main():
    """Train an agent with custom reward function."""
    print("Training agent with custom reward function...")
    
    # Create trainer with custom reward
    trainer = Trainer(
        reward_function=CustomReward(),
        hyperparams="aggressive",  # Use aggressive hyperparameters
        eval_freq=2500
    )
    
    # Train the agent
    model, model_dir = trainer.train(
        total_timesteps=15000,
        model_name="custom_reward_agent"
    )
    
    print(f"\nTraining complete with custom reward!")
    print(f"Model saved in: {model_dir}")

if __name__ == "__main__":
    main()