"""
Example demonstrating immediate rewards vs delayed rewards in Big Two RL.

This example shows how to:
1. Create custom reward functions with the new API
2. Compare immediate vs delayed feedback training
3. Use episode bonuses for overall performance incentives

Key improvements with immediate rewards:
- Faster learning from individual game outcomes
- Better signal-to-noise ratio
- Maintained episode-level performance tracking
"""

import numpy as np
from bigtwo_rl.training import Trainer
from bigtwo_rl.training.rewards import BaseReward
from bigtwo_rl.evaluation import Evaluator


class FastLearningReward(BaseReward):
    """Custom reward designed for fast learning with clear signals."""

    def game_reward(self, winner_player, player_idx, cards_left, all_cards_left=None):
        """
        Immediate reward after each game:
        - Clear win signal: +2.0
        - Progressive loss penalty based on cards remaining
        """
        if player_idx == winner_player:
            return 2.0  # Strong positive signal for winning
        else:
            # Progressive penalty - worse to lose with many cards
            if cards_left >= 8:
                return -1.0  # Bad performance
            elif cards_left >= 4:
                return -0.5  # Mediocre performance
            else:
                return -0.1  # Close game, small penalty

    def episode_bonus(self, games_won, total_games, avg_cards_left):
        """
        Episode bonus for consistent good performance:
        - High win rate gets bonus
        - Low average cards when losing gets bonus
        """
        win_rate = games_won / total_games if total_games > 0 else 0

        # Win rate bonus
        if win_rate > 0.8:
            bonus = 1.0
        elif win_rate > 0.6:
            bonus = 0.5
        else:
            bonus = 0

        # Add bonus for keeping card count low when losing
        if avg_cards_left < 3.0:
            bonus += 0.3
        elif avg_cards_left < 6.0:
            bonus += 0.1

        return bonus


class DelayedOnlyReward(BaseReward):
    """Traditional delayed reward for comparison."""

    def game_reward(self, winner_player, player_idx, cards_left, all_cards_left=None):
        """No immediate rewards - only episode end matters."""
        return 0.0  # No immediate feedback

    def episode_bonus(self, games_won, total_games, avg_cards_left):
        """All reward comes at episode end."""
        win_rate = games_won / total_games if total_games > 0 else 0

        if win_rate > 0.6:
            return 5.0  # Large delayed reward
        else:
            return -2.0 * (total_games - games_won)  # Penalty for losses


def compare_immediate_vs_delayed_training():
    """Compare training performance between immediate and delayed rewards."""

    print("=== Immediate vs Delayed Reward Training Comparison ===\n")

    # Configuration for fast comparison
    train_steps = 15000
    eval_games = 50

    print("Training both agents for", train_steps, "steps...\n")

    # Train with immediate rewards
    print("1. Training with Immediate Rewards (FastLearningReward):")
    immediate_trainer = Trainer(
        reward_function=FastLearningReward(),
        hyperparams="fast_experimental",  # Use fast hyperparams for demo
    )
    immediate_model, immediate_dir = immediate_trainer.train(
        total_timesteps=train_steps, model_name="immediate_reward_demo"
    )
    print(f"   Model saved to: {immediate_dir}")

    # Train with delayed rewards
    print("\n2. Training with Delayed Rewards (DelayedOnlyReward):")
    delayed_trainer = Trainer(
        reward_function=DelayedOnlyReward(), hyperparams="fast_experimental"
    )
    delayed_model, delayed_dir = delayed_trainer.train(
        total_timesteps=train_steps, model_name="delayed_reward_demo"
    )
    print(f"   Model saved to: {delayed_dir}")

    print(f"\n=== Evaluation Results ({eval_games} games each) ===")

    # Evaluate both models
    evaluator = Evaluator(num_games=eval_games)

    print("\n1. Immediate Reward Agent Performance:")
    immediate_results = evaluator.evaluate_model(f"{immediate_dir}/final_model")
    print(f"   Win Rate: {immediate_results['win_rate']:.1%}")
    print(f"   Avg Cards Left: {immediate_results['avg_cards_left']:.1f}")

    print("\n2. Delayed Reward Agent Performance:")
    delayed_results = evaluator.evaluate_model(f"{delayed_dir}/final_model")
    print(f"   Win Rate: {delayed_results['win_rate']:.1%}")
    print(f"   Avg Cards Left: {delayed_results['avg_cards_left']:.1f}")

    # Compare results
    print("\n=== Comparison Summary ===")
    immediate_wr = immediate_results["win_rate"]
    delayed_wr = delayed_results["win_rate"]

    if immediate_wr > delayed_wr:
        improvement = (immediate_wr - delayed_wr) / delayed_wr * 100
        print(f"✓ Immediate rewards achieved {improvement:.1f}% better win rate")
    else:
        print(f"⚠ Delayed rewards performed better in this run")

    print(f"\nImmediate reward benefits:")
    print(f"- Faster feedback after each game completion")
    print(f"- Better learning signals for individual game performance")
    print(f"- Episode bonuses maintain longer-term incentives")

    return immediate_results, delayed_results


def demonstrate_custom_reward_creation():
    """Show how to create and use custom reward functions."""

    print("\n=== Custom Reward Function Creation ===\n")

    class AggressiveWinnerReward(BaseReward):
        """Reward function that heavily incentivizes winning quickly."""

        def game_reward(
            self, winner_player, player_idx, cards_left, all_cards_left=None
        ):
            if player_idx == winner_player:
                return 5.0  # Very high win reward
            else:
                return -0.05 * cards_left  # Small loss penalty

        def episode_bonus(self, games_won, total_games, avg_cards_left):
            # Huge bonus for perfect episodes
            if games_won == total_games:
                return 3.0
            return 0

    class ConservativeReward(BaseReward):
        """Reward function focused on minimizing cards when losing."""

        def game_reward(
            self, winner_player, player_idx, cards_left, all_cards_left=None
        ):
            if player_idx == winner_player:
                return 1.0
            else:
                # Heavy penalty for many cards
                return -0.3 * cards_left

        def episode_bonus(self, games_won, total_games, avg_cards_left):
            # Reward low card average even with moderate wins
            if avg_cards_left < 4.0:
                return 1.0
            elif avg_cards_left < 7.0:
                return 0.3
            return 0

    print("Created two custom reward functions:")
    print("1. AggressiveWinnerReward - High win rewards, small loss penalties")
    print("2. ConservativeReward - Focus on minimizing cards when losing")

    print("\nBoth can be used with:")
    print("trainer = Trainer(reward_function=AggressiveWinnerReward())")
    print("trainer = Trainer(reward_function=ConservativeReward())")

    # Test the reward functions
    print("\n--- Reward Function Testing ---")
    aggressive = AggressiveWinnerReward()
    conservative = ConservativeReward()

    # Win scenario
    print(f"Win reward - Aggressive: {aggressive.game_reward(0, 0, 0):.1f}")
    print(f"Win reward - Conservative: {conservative.game_reward(0, 0, 0):.1f}")

    # Loss scenario (5 cards)
    print(f"Loss reward (5 cards) - Aggressive: {aggressive.game_reward(1, 0, 5):.1f}")
    print(
        f"Loss reward (5 cards) - Conservative: {conservative.game_reward(1, 0, 5):.1f}"
    )


def main():
    """Main demonstration function."""
    print("Big Two RL - Immediate Rewards Example")
    print("=" * 50)

    # Demonstrate custom reward creation
    demonstrate_custom_reward_creation()

    # Compare immediate vs delayed rewards
    try:
        immediate_results, delayed_results = compare_immediate_vs_delayed_training()

        print(f"\n✓ Training comparison completed successfully!")
        print(f"Check the generated model directories for saved agents.")

    except Exception as e:
        print(f"Training comparison failed: {e}")
        print(
            "This might be due to training time - try reducing train_steps for faster demo"
        )

    print(f"\n--- Key Takeaways ---")
    print(f"1. Immediate rewards provide feedback after each game")
    print(f"2. Episode bonuses maintain longer-term performance incentives")
    print(f"3. Custom reward functions are easy to create with BaseReward")
    print(f"4. The new API allows flexible reward experimentation")


if __name__ == "__main__":
    main()
