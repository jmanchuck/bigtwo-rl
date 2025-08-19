#!/usr/bin/env python3
"""
Example demonstrating the new explicit observation configuration API.

This shows how to use explicit observation configurations instead of magic strings.
"""

from bigtwo_rl.training import Trainer
from bigtwo_rl.training.rewards import DefaultReward
from bigtwo_rl.training.hyperparams import FastExperimentalConfig
from bigtwo_rl import (
    minimal_observation,
    standard_observation,
    memory_enhanced_observation,
    strategic_observation,
    ObservationBuilder,
)


def main():
    print("🔧 Big Two RL - Explicit Observation Configuration API")
    print("=" * 60)

    # Example 1: Using preset observation functions (type-safe)
    print("\n1. Preset Observation Configurations")
    print("-" * 40)

    configs = [
        ("Minimal", minimal_observation()),  # 56 features
        ("Standard", standard_observation()),  # 109 features
        ("Memory Enhanced", memory_enhanced_observation()),  # 161 features
        ("Strategic", strategic_observation()),  # 325+ features
    ]

    for name, config in configs:
        print(f"• {name:15}: {config._total_size:3d} features")

        # Each config can be used directly in training
        trainer = Trainer(
            reward_function=DefaultReward(),
            hyperparams=FastExperimentalConfig(),
            observation_config=config,
        )
        # trainer.train(...) would work here

    print("\n✅ All configurations are explicit ObservationConfig instances")
    print("✅ No magic strings - IDE autocomplete works")
    print("✅ Type safety - errors caught at write-time")

    # Example 2: Custom observation builder (flexible)
    print("\n2. Custom Observation Builder")
    print("-" * 40)

    # Build custom configurations using fluent interface
    card_counter_config = (
        ObservationBuilder()
        .minimal()  # Start minimal
        .with_last_play()  # See current trick
        .with_card_memory()  # Track played cards
        .with_power_card_tracking()  # Focus on key cards
        .build()
    )

    print(f"• Card Counter Config: {card_counter_config._total_size} features")
    print("  - Own hand + hand sizes")
    print("  - Last play + play exists flag")
    print("  - Memory of all played cards")
    print("  - Power card tracking (Aces, 2s)")

    opponent_analyst_config = (
        ObservationBuilder()
        .standard()  # Standard base
        .with_opponent_modeling()  # Focus on opponents
        .with_trick_history()  # Who wins tricks
        .with_game_context()  # Game phase awareness
        .build()
    )

    print(
        f"\n• Opponent Analyst Config: {opponent_analyst_config._total_size} features"
    )
    print("  - Standard observation space")
    print("  - Pass history and play patterns")
    print("  - Recent trick winners")
    print("  - Game phase and turn position")

    # Example 3: Comparison with old API (NO LONGER WORKS)
    print("\n3. New vs Old API")
    print("-" * 40)

    print("❌ OLD (strings - no longer supported):")
    print('   trainer = Trainer(observation_config="minimal")')
    print('   trainer = Trainer(observation_config="strategic")')

    print("\n✅ NEW (explicit classes - type-safe):")
    print("   trainer = Trainer(observation_config=minimal_observation())")
    print("   trainer = Trainer(observation_config=strategic_observation())")

    print("\n📝 Key Benefits:")
    print("• Import statements reveal available options")
    print("• IDE autocomplete helps discover configurations")
    print("• Type checking catches errors before runtime")
    print("• No guessing what string constants exist")
    print("• Clear, explicit, self-documenting code")

    # Example 4: Show feature counts for research
    print("\n4. Feature Count Analysis")
    print("-" * 40)

    all_configs = [
        ("Minimal", minimal_observation()),
        ("Standard", standard_observation()),
        ("Memory Enhanced", memory_enhanced_observation()),
        ("Strategic", strategic_observation()),
    ]

    print("Configuration          Features  Training Speed")
    print("-" * 50)
    for name, config in all_configs:
        speed = {
            56: "Fastest",
            109: "Fast",
            161: "Moderate",
        }.get(config._total_size, "Slower")

        print(f"{name:20} {config._total_size:8d}  {speed}")

    print(f"\nChoose based on:")
    print(f"• Minimal: Fastest training, basic gameplay")
    print(f"• Standard: Good balance, proven effective")
    print(f"• Memory Enhanced: Better strategy, moderate speed")
    print(f"• Strategic: Maximum intelligence, slower training")


if __name__ == "__main__":
    main()
