#!/usr/bin/env python3
"""
Example: Custom Observation Configurations

This example demonstrates how to train models with different observation spaces
and then pit them against each other in tournaments to see which information
gives the biggest advantage.
"""

from bigtwo_rl.training import Trainer
from bigtwo_rl.training.rewards import DefaultReward, ProgressiveReward, RankingReward
from bigtwo_rl.training.hyperparams import FastExperimentalConfig
from bigtwo_rl.core.observation_builder import ObservationBuilder
from bigtwo_rl.agents import PPOAgent, RandomAgent, GreedyAgent
from bigtwo_rl.evaluation import Tournament


def main():
    print("🃏 Big Two RL - Custom Observation Example")
    print("=" * 50)

    # 1. Train a "blind" model (minimal information)
    print("\n1. Training minimal-vision model (only hand + hand sizes)...")
    blind_config = ObservationBuilder().minimal().build()

    trainer_blind = Trainer(
        reward_function=DefaultReward(),
        hyperparams=FastExperimentalConfig(),
        observation_config=blind_config,
    )

    print(f"   Observation space size: {blind_config._total_size} features")
    model_blind, dir_blind = trainer_blind.train(
        total_timesteps=10000, model_name="blind_agent"
    )
    print(f"   Model saved to: {dir_blind}")

    # 2. Train a "memory" model (with card tracking)
    print("\n2. Training memory-enhanced model...")
    memory_config = (
        ObservationBuilder()
        .standard()  # Start with standard
        .with_card_memory()  # Add card memory
        .with_remaining_deck()  # Track what's left
        .with_power_card_tracking()  # Track Aces and 2s
        .build()
    )

    trainer_memory = Trainer(
        reward_function=ProgressiveReward(),
        hyperparams=FastExperimentalConfig(),
        observation_config=memory_config,
    )

    print(f"   Observation space size: {memory_config._total_size} features")
    model_memory, dir_memory = trainer_memory.train(
        total_timesteps=10000, model_name="memory_agent"
    )
    print(f"   Model saved to: {dir_memory}")

    # 3. Train a "strategic" model (full information)
    print("\n3. Training strategic model (full information)...")
    strategic_config = (
        ObservationBuilder()
        .strategic()  # All advanced features
        .build()
    )

    trainer_strategic = Trainer(
        reward_function=RankingReward(),
        hyperparams=FastExperimentalConfig(),
        observation_config=strategic_config,
    )

    print(f"   Observation space size: {strategic_config._total_size} features")
    model_strategic, dir_strategic = trainer_strategic.train(
        total_timesteps=10000, model_name="strategic_agent"
    )
    print(f"   Model saved to: {dir_strategic}")

    # 4. Create agents with their respective observation configs
    print("\n4. Setting up tournament...")
    agents = [
        PPOAgent(
            model_path=f"{dir_blind}/best_model",
            name="Blind-Agent",
            observation_config=blind_config,
        ),
        PPOAgent(
            model_path=f"{dir_memory}/best_model",
            name="Memory-Agent",
            observation_config=memory_config,
        ),
        PPOAgent(
            model_path=f"{dir_strategic}/best_model",
            name="Strategic-Agent",
            observation_config=strategic_config,
        ),
        RandomAgent("Random-Baseline"),
    ]

    # 5. Run tournament to compare performance
    print("\n5. Running tournament (this may take a moment)...")
    tournament = Tournament(agents)
    results = tournament.run(num_games=50)  # Quick tournament for demo

    print("\n" + "=" * 60)
    print("🏆 TOURNAMENT RESULTS")
    print("=" * 60)
    print(results["tournament_summary"])

    print("\n" + "=" * 60)
    print("📊 ANALYSIS")
    print("=" * 60)
    print("Information Levels:")
    print(f"• Blind Agent:     {blind_config._total_size:3d} features (minimal info)")
    print(f"• Memory Agent:    {memory_config._total_size:3d} features (card tracking)")
    print(f"• Strategic Agent: {strategic_config._total_size:3d} features (full info)")
    print(f"• Random Baseline:   109 features (standard)")

    print("\nThis demonstrates how different observation configurations")
    print("can be trained and compared to understand the value of")
    print("different types of game information!")


def demonstrate_custom_builder():
    """Show how to build completely custom observation configs."""
    print("\n" + "=" * 60)
    print("🔧 CUSTOM BUILDER EXAMPLES")
    print("=" * 60)

    # Example 1: Card counter agent
    card_counter = (
        ObservationBuilder()
        .minimal()  # Start minimal
        .with_last_play()  # Need to see plays
        .with_card_memory()  # Track all played cards
        .with_power_card_tracking()  # Focus on key cards
        .build()
    )

    print(f"Card Counter Config: {card_counter._total_size} features")
    print("  - Own hand + hand sizes")
    print("  - Last play + play exists flag")
    print("  - Memory of all played cards")
    print("  - Power card tracking (Aces, 2s)")

    # Example 2: Opponent modeling specialist
    opponent_analyst = (
        ObservationBuilder()
        .standard()  # Standard base
        .with_opponent_modeling()  # Focus on opponents
        .with_trick_history()  # Who wins tricks
        .with_game_context()  # Game phase awareness
        .build()
    )

    print(f"\nOpponent Analyst Config: {opponent_analyst._total_size} features")
    print("  - Standard observation space")
    print("  - Pass history and play patterns")
    print("  - Recent trick winners")
    print("  - Game phase and turn position")

    # Example 3: Minimal but focused
    focused_minimal = (
        ObservationBuilder()
        .minimal()
        .with_power_card_tracking()  # Only add power cards
        .build()
    )

    print(f"\nFocused Minimal Config: {focused_minimal._total_size} features")
    print("  - Just hand, hand sizes, and power card status")
    print("  - Fastest training, focused on key strategic cards")


if __name__ == "__main__":
    main()
    demonstrate_custom_builder()
