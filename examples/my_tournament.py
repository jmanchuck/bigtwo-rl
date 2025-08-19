#!/usr/bin/env python3
"""Quick tournament setup - just update the model paths and run!"""

from bigtwo_rl.agents import RandomAgent, load_ppo_agent
from bigtwo_rl.evaluation import Tournament

# ===== UPDATE THESE PATHS TO YOUR TRAINED MODELS =====
MODEL1_PATH = "./models/my_first_model/best_model"  # Replace with your first model
MODEL2_PATH = "./models/my_second_model/best_model"  # Replace with your second model


def main():
    print("Setting up tournament with 2 PPO agents + 2 random agents...")

    agents = [
        # Your trained agents
        load_ppo_agent(MODEL1_PATH, "My-Agent-1"),
        load_ppo_agent(MODEL2_PATH, "My-Agent-2"),
        # Baseline random agents
        RandomAgent("Random-1"),
        RandomAgent("Random-2"),
    ]

    print(f"Agents: {[a.name for a in agents]}")

    # Run tournament with multiprocessing (adjust num_games as needed)
    # n_processes=None for auto-detection, or specify a number like n_processes=4
    tournament = Tournament(agents, n_processes=None)
    results = tournament.run(
        num_games=200
    )  # Increased for better multiprocessing benefit

    print(results["tournament_summary"])


if __name__ == "__main__":
    main()
