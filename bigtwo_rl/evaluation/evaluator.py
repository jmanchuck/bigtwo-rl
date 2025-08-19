#!/usr/bin/env python3
"""Evaluate trained agent against baselines."""

from ..agents import RandomAgent, GreedyAgent, PPOAgent, BaseAgent
from .tournament import play_four_player_series


class Evaluator:
    """High-level evaluator for Big Two agents."""

    def __init__(self, num_games: int = 100, n_processes=None):
        """
        Initialize evaluator.

        Args:
            num_games: Number of games per evaluation
            n_processes: Number of processes for parallel execution (None = auto)
        """
        self.num_games = num_games
        self.n_processes = n_processes

    def evaluate_agent(self, agent: BaseAgent, baselines: bool = True) -> dict:
        """
        Evaluate an agent in 4-player series against three opponents.

        Args:
            agent: Agent to evaluate
            baselines: If True, opponents are a mix of Random and Greedy. If False, caller must supply fully formed list elsewhere.

        Returns:
            Dict with evaluation results
        """
        # Always construct a 4-player table: [agent] + 3 opponents
        if not baselines:
            raise ValueError(
                "evaluate_agent requires baselines=True to auto-generate three opponents"
            )

        # Create a mix of random/greedy opponents (2 Random, 1 Greedy by default)
        opponents = [
            RandomAgent("Random-1"),
            RandomAgent("Random-2"),
            GreedyAgent("Greedy"),
        ]

        table_agents = [agent] + opponents
        return play_four_player_series(table_agents, self.num_games, self.n_processes)

    def evaluate_model(self, model_path: str, model_name: str = None) -> dict:
        """
        Evaluate a trained PPO model against baselines.

        Args:
            model_path: Path to saved PPO model
            model_name: Name for the model (defaults to path)

        Returns:
            Dict with evaluation results
        """
        if model_name is None:
            model_name = f"PPO-{model_path.split('/')[-1]}"

        agent = PPOAgent(model_path, model_name)
        return self.evaluate_agent(agent)

    def compare_models(self, model_paths: list, model_names: list = None) -> dict:
        """
        Compare multiple trained models against each other and baselines.

        Args:
            model_paths: List of paths to saved PPO models
            model_names: Optional list of names for models

        Returns:
            Dict with tournament results
        """
        if model_names is None:
            model_names = [f"PPO-{path.split('/')[-1]}" for path in model_paths]

        agents = [PPOAgent(path, name) for path, name in zip(model_paths, model_names)]

        # For compare, run every 4-player combination among models plus baselines if needed
        # Here, we compare each model in a fixed table against three baselines for consistency
        results = {}
        for agent in agents:
            opponents = [
                RandomAgent("Random-1"),
                RandomAgent("Random-2"),
                GreedyAgent("Greedy"),
            ]
            table_agents = [agent] + opponents
            results[agent.name] = play_four_player_series(
                table_agents, self.num_games, self.n_processes
            )
        return {"per_agent_series": results}


def evaluate_agent(model_path, num_games=100):
    """Evaluate a PPO model in a 4-player series against three baselines."""
    agent = PPOAgent(model_path, name=f"PPO-{model_path.split('/')[-1]}")
    opponents = [
        RandomAgent("Random-1"),
        RandomAgent("Random-2"),
        GreedyAgent("Greedy"),
    ]
    table_agents = [agent] + opponents
    return play_four_player_series(table_agents, num_games)


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "./models/best_model"
    results = evaluate_agent(model_path)
