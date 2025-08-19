#!/usr/bin/env python3
"""
Example demonstrating the new explicit class-based API.

This shows the cleaner, type-safe way to configure training:
- No magic strings
- IDE autocomplete support
- Clear imports show exactly what's being used
- Compile-time validation instead of runtime errors
"""

from bigtwo_rl.training import Trainer
from bigtwo_rl.training.rewards import DefaultReward, SparseReward, ScoreMarginReward
from bigtwo_rl.training.hyperparams import (
    DefaultConfig,
    AggressiveConfig,
    ConservativeConfig,
)


def main():
    print("🃏 Big Two RL - New Explicit Class-Based API")
    print("=" * 50)

    # Example 1: Basic training with default settings
    print("\n1. Basic Training (Default Config + Default Reward)")
    trainer = Trainer(reward_function=DefaultReward(), hyperparams=DefaultConfig())

    # Example 2: Aggressive training with sparse rewards
    print("\n2. Fast Experimental Training")
    fast_trainer = Trainer(
        reward_function=SparseReward(), hyperparams=AggressiveConfig()
    )

    # Example 3: Conservative training with score margin rewards
    print("\n3. Conservative Training")
    conservative_trainer = Trainer(
        reward_function=ScoreMarginReward(), hyperparams=ConservativeConfig()
    )

    print("\n✅ Key Benefits of New API:")
    print("• Type-safe: IDE catches errors at write-time")
    print("• Discoverable: Import statements show available options")
    print("• Clear: No magic strings, explicit class instantiation")
    print("• Extensible: Easy to add custom rewards/configs")

    print("\n📝 Usage Pattern:")
    print("1. Import the specific classes you want to use")
    print("2. Instantiate them explicitly: DefaultReward(), AggressiveConfig()")
    print("3. Pass instances to Trainer constructor")
    print("4. No more guessing what string constants are available!")

    # Show what the classes provide
    print(f"\n🔧 Available Reward Classes:")
    print(f"• DefaultReward - Balanced win/loss rewards")
    print(f"• SparseReward - Simple win/loss only")
    print(f"• ScoreMarginReward - Card advantage based")
    print(f"• + AggressivePenaltyReward, ProgressiveReward, RankingReward")

    print(f"\n⚙️  Available Hyperparameter Classes:")
    print(f"• DefaultConfig - Balanced training settings")
    print(f"• AggressiveConfig - Fast, less stable training")
    print(f"• ConservativeConfig - Slow, stable training")
    print(f"• FastExperimentalConfig - Quick testing")


if __name__ == "__main__":
    main()
