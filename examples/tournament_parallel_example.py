#!/usr/bin/env python3
"""Example of running parallel tournaments with Big Two RL agents."""

import time
from bigtwo_rl.agents import RandomAgent, GreedyAgent, PPOAgent
from bigtwo_rl.evaluation import Tournament, Evaluator


def main():
    """Run parallel tournament examples."""

    print("Big Two RL - Parallel Tournament Example")
    print("=" * 50)

    # Create agents for tournament
    agents = [
        RandomAgent("Random-1"),
        RandomAgent("Random-2"),
        GreedyAgent("Greedy-1"),
        GreedyAgent("Greedy-2"),
    ]

    # Example 1: Basic parallel tournament
    print("\n1. Basic Parallel Tournament")
    print("-" * 30)

    # Create tournament with automatic multiprocessing
    tournament = Tournament(agents, n_processes=None)  # Auto-detect CPU cores

    start_time = time.time()
    results = tournament.run(num_games=500)
    elapsed = time.time() - start_time

    print(f"Tournament completed in {elapsed:.2f} seconds")
    print(results["tournament_summary"])

    # Example 2: Compare sequential vs parallel
    print("\n2. Performance Comparison")
    print("-" * 30)

    # Sequential tournament
    tournament_seq = Tournament(agents, n_processes=1)
    start_time = time.time()
    tournament_seq.run(num_games=200)
    seq_time = time.time() - start_time

    # Parallel tournament
    tournament_par = Tournament(agents, n_processes=4)
    start_time = time.time()
    tournament_par.run(num_games=200)
    par_time = time.time() - start_time

    print(f"Sequential (1 process): {seq_time:.2f}s")
    print(f"Parallel (4 processes): {par_time:.2f}s")
    print(f"Speedup: {seq_time/par_time:.2f}x")

    # Example 3: Evaluating a model (if available)
    print("\n3. Parallel Model Evaluation")
    print("-" * 30)

    try:
        # Try to load a trained model
        ppo_agent = PPOAgent(model_path="./models/best_model", name="PPO-Best")

        # Create evaluator with parallel processing
        evaluator = Evaluator(num_games=300, n_processes=4)

        start_time = time.time()
        eval_results = evaluator.evaluate_agent(ppo_agent)
        eval_time = time.time() - start_time

        print(f"Evaluation completed in {eval_time:.2f} seconds")
        print(f"PPO Agent win rate: {eval_results['win_rates']['PPO-Best']:.1%}")

    except (FileNotFoundError, Exception):
        print("No trained model found. Skipping model evaluation.")
        print("Train a model first using: uv run python examples/train_agent.py")

    # Example 4: Custom process control
    print("\n4. Custom Process Control")
    print("-" * 30)

    agents_small = [
        RandomAgent("R1"),
        GreedyAgent("G1"),
        RandomAgent("R2"),
        GreedyAgent("G2"),
    ]

    for n_proc in [1, 2, 4]:
        tournament = Tournament(agents_small, n_processes=n_proc)
        start_time = time.time()
        tournament.run(num_games=100)
        elapsed = time.time() - start_time
        print(f"  {n_proc} process(es): {elapsed:.2f}s ({100/elapsed:.1f} games/sec)")


if __name__ == "__main__":
    main()
