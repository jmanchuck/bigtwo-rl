"""Play Big Two against trained AI agents using Tournament framework."""

import sys
import os
from typing import cast
from bigtwo_rl.agents import HumanAgent, PPOAgent, GreedyAgent

try:
    from bigtwo_rl.evaluation import Tournament  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    Tournament = None  # type: ignore[assignment]


def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/play_vs_agent.py <MODEL_PATH_OR_GREEDY>")
        print("  MODEL_PATH: Path to trained model directory or .zip file")
        print("  greedy: Play against 3 greedy agents instead of AI models")
        print("Examples:")
        print("  python examples/play_vs_agent.py ./models/my_model/best_model.zip")
        print("  python examples/play_vs_agent.py greedy")
        sys.exit(1)

    opponent_type = sys.argv[1]

    # Check if user wants to play against greedy agents
    if opponent_type.lower() == "greedy":
        use_greedy = True
        model_path = None
    else:
        use_greedy = False
        model_path = opponent_type

        # Validate model path
        if os.path.isdir(model_path):
            # Look for model files in directory
            candidates = [
                os.path.join(model_path, "best_model.zip"),
                os.path.join(model_path, "final_model.zip"),
            ]
            actual_model_path = None
            for candidate in candidates:
                if os.path.isfile(candidate):
                    actual_model_path = candidate
                    break

            if actual_model_path is None:
                # Try any .zip file in directory
                zip_files = [f for f in os.listdir(model_path) if f.endswith(".zip")]
                if zip_files:
                    actual_model_path = os.path.join(model_path, sorted(zip_files)[0])

            if actual_model_path is None:
                print(f"❌ No model file found in directory: {model_path}")
                print("Expected 'best_model.zip' or 'final_model.zip'")
                sys.exit(1)
            model_path = actual_model_path
        elif not os.path.isfile(model_path):
            print(f"❌ Model file not found: {model_path}")
            sys.exit(1)

    print("🃏 BIG TWO: Human vs AI Agents")
    print("=" * 50)
    print("🎮 Game Rules:")
    print("  • Card Ranks: 3 < 4 < 5 < 6 < 7 < 8 < 9 < T < J < Q < K < A < 2")
    print("  • Suits: Diamonds < Clubs < Hearts < Spades")
    print("  • Play Types: Singles, Pairs, Trips, 5-card combinations")
    print("  • Goal: Be first to play all your cards!")
    print()
    print("💡 Controls:")
    print("  • Enter move number (0, 1, 2, etc.) to select your play")
    print("  • Type 'quit' or 'q' to exit anytime")
    print("=" * 50)

    try:
        # Create agents
        human = HumanAgent("Human")

        if use_greedy:
            agent1 = GreedyAgent("Greedy-1")
            agent2 = GreedyAgent("Greedy-2")
            agent3 = GreedyAgent("Greedy-3")
            print("✅ Playing against 3 Greedy agents")
            opponent_type_display = "Greedy"
        else:
            agent1 = PPOAgent(cast(str, model_path), "AI-Agent-1")
            agent2 = PPOAgent(cast(str, model_path), "AI-Agent-2")
            agent3 = PPOAgent(cast(str, model_path), "AI-Agent-3")
            print(f"✅ Loaded AI model from: {model_path}")
            opponent_type_display = "AI"

        print("🎯 Starting game with you as Player 0...")
        print()

        # Create tournament (single game)
        if Tournament is None:
            print("❌ Tournament module not available in this build.")
            sys.exit(1)
        tournament = Tournament([human, agent1, agent2, agent3])

        # Run the game
        results = tournament.run(num_games=1)

        # Display final results
        print("\n" + "=" * 60)
        print("🏆 FINAL RESULTS")
        print("=" * 60)

        # Get the first (and only) matchup result for a single game
        matchup = results["matchup_results"][0]

        # Create player position mapping
        agent_names = [human.name, agent1.name, agent2.name, agent3.name]
        player_labels = [
            "Human",
            f"{opponent_type_display} 1",
            f"{opponent_type_display} 2",
            f"{opponent_type_display} 3",
        ]

        # Find winner and display
        winner_found = False
        for player_name, wins in matchup["wins"].items():
            if wins > 0:
                player_idx = agent_names.index(player_name)
                print(f"🥇 WINNER: {player_labels[player_idx]}")
                winner_found = True
                break

        if not winner_found:
            print("🤔 No winner found (this should not happen)")

        print("\nCards remaining:")
        # For a single game, get the actual cards remaining from the last game
        # The cards_left_by_game contains lists of [player0_cards, player1_cards, player2_cards, player3_cards]
        if "cards_left_by_game" in matchup and matchup["cards_left_by_game"]:
            final_cards = matchup["cards_left_by_game"][-1]  # Last (and only) game
            for i, cards in enumerate(final_cards):
                print(f"  {player_labels[i]}: {cards} cards")
        else:
            # Fallback to averaged data if individual game data not available
            avg_cards = matchup["avg_cards_left"]
            for i, agent_name in enumerate(agent_names):
                if agent_name in avg_cards:
                    cards = avg_cards[agent_name]
                    print(f"  {player_labels[i]}: {cards:.0f} cards")

        print("=" * 60)
        print("👋 Thanks for playing Big Two!")

    except FileNotFoundError as e:
        print(f"❌ Model loading failed: {e}")
        print("Make sure the model file exists and is a valid PPO model.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Game interrupted. Thanks for playing!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Please check your model file and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
