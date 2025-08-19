#!/usr/bin/env python3
"""Benchmark multiprocessing performance improvements for tournaments."""

import time
import multiprocessing as mp
from bigtwo_rl.agents import RandomAgent, GreedyAgent
from bigtwo_rl.evaluation import Tournament, play_four_player_series


def benchmark_tournament_performance():
    """Compare sequential vs parallel tournament performance."""

    # Create test agents
    agents = [
        RandomAgent("Random-1"),
        RandomAgent("Random-2"),
        GreedyAgent("Greedy-1"),
        GreedyAgent("Greedy-2"),
    ]

    num_games = 200
    print(f"Benchmarking tournament with {num_games} games...")
    print(f"Available CPU cores: {mp.cpu_count()}")
    print("=" * 60)

    # Test sequential execution
    print("1. Sequential execution (n_processes=1)")
    start_time = time.time()
    sequential_results = play_four_player_series(agents, num_games, n_processes=1)
    sequential_time = time.time() - start_time
    print(f"   Time: {sequential_time:.2f} seconds")
    print(f"   Games/sec: {num_games/sequential_time:.1f}")
    print()

    # Test parallel execution with different process counts
    for n_proc in [2, 4, mp.cpu_count()]:
        if n_proc > mp.cpu_count():
            continue

        print(f"2. Parallel execution ({n_proc} processes)")
        start_time = time.time()
        parallel_results = play_four_player_series(
            agents, num_games, n_processes=n_proc
        )
        parallel_time = time.time() - start_time
        speedup = sequential_time / parallel_time

        print(f"   Time: {parallel_time:.2f} seconds")
        print(f"   Games/sec: {num_games/parallel_time:.1f}")
        print(f"   Speedup: {speedup:.2f}x")
        print()

    # Verify results are consistent
    print("3. Results verification:")
    print(f"   Sequential wins: {sequential_results['wins']}")
    print(f"   Parallel wins: {parallel_results['wins']}")
    print(
        f"   Win rates match: {abs(sum(sequential_results['win_rates'].values()) - sum(parallel_results['win_rates'].values())) < 0.01}"
    )


def benchmark_tournament_class():
    """Test Tournament class with multiprocessing."""

    agents = [
        RandomAgent("Random-1"),
        RandomAgent("Random-2"),
        GreedyAgent("Greedy-1"),
        GreedyAgent("Greedy-2"),
    ]

    print("\nTesting Tournament class:")
    print("=" * 40)

    # Sequential tournament
    tournament_seq = Tournament(agents, n_processes=1)
    start_time = time.time()
    results_seq = tournament_seq.run(num_games=100)
    seq_time = time.time() - start_time

    # Parallel tournament
    tournament_par = Tournament(agents, n_processes=4)
    start_time = time.time()
    results_par = tournament_par.run(num_games=100)
    par_time = time.time() - start_time

    print(f"Sequential: {seq_time:.2f}s")
    print(f"Parallel: {par_time:.2f}s")
    print(f"Tournament speedup: {seq_time/par_time:.2f}x")

    # Show tournament summary
    print("\nTournament Results:")
    print(results_par["tournament_summary"])


def profile_game_performance():
    """Profile individual game vs batch performance."""

    agents = [
        RandomAgent("R1"),
        RandomAgent("R2"),
        GreedyAgent("G1"),
        GreedyAgent("G2"),
    ]

    print("\nGame Performance Profile:")
    print("=" * 40)

    # Single games timing
    from bigtwo_rl.core.rl_wrapper import BigTwoRLWrapper
    from bigtwo_rl.evaluation.tournament import play_single_game

    env = BigTwoRLWrapper(num_players=4, games_per_episode=1)

    # Time 50 individual games
    start_time = time.time()
    for _ in range(50):
        play_single_game(agents, env)
    single_game_time = (time.time() - start_time) / 50

    print(f"Average single game time: {single_game_time*1000:.1f}ms")
    print(f"Theoretical max games/sec: {1/single_game_time:.0f}")

    # Compare with actual batch performance
    start_time = time.time()
    play_four_player_series(agents, 50, n_processes=1)
    batch_time = time.time() - start_time

    print(
        f"Actual 50 games (sequential): {batch_time:.2f}s ({batch_time/50*1000:.1f}ms per game)"
    )
    print(f"Overhead per game: {(batch_time/50 - single_game_time)*1000:.1f}ms")


if __name__ == "__main__":
    print("Big Two RL - Multiprocessing Performance Benchmark")
    print("=" * 60)

    benchmark_tournament_performance()
    benchmark_tournament_class()
    profile_game_performance()

    print("\nBenchmark complete!")
