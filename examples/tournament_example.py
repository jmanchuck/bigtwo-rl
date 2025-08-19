#!/usr/bin/env python3
"""Simple example: Run a tournament between multiple agents."""

from bigtwo_rl.agents import RandomAgent, GreedyAgent, PPOAgent
from bigtwo_rl.evaluation import Tournament

def main():
    """Run a tournament with available agents."""
    print("Setting up tournament...")
    
    # Create baseline agents
    agents = [
        RandomAgent("Random"),
        GreedyAgent("Greedy"),
    ]
    
    # Try to add trained agents if they exist
    model_paths = [
        "./models/example_agent/best_model",
        "./models/best_model.zip",  # Legacy model
    ]
    
    for model_path in model_paths:
        try:
            model_name = f"PPO-{model_path.split('/')[-1].replace('.zip', '')}"
            agents.append(PPOAgent(model_path, model_name))
            print(f"Added trained agent: {model_name}")
        except Exception as e:
            print(f"Could not load model {model_path}: {e}")
    
    if len(agents) < 2:
        print("Need at least 2 agents for tournament!")
        return
        
    print(f"\nRunning tournament with {len(agents)} agents:")
    for agent in agents:
        print(f"  - {agent.name}")
    
    # Create tournament and run it
    tournament = Tournament(agents)
    results = tournament.run_round_robin(num_games=50)
    
    # Print results
    print("\n" + results["tournament_summary"])

if __name__ == "__main__":
    main()