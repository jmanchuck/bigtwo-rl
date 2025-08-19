#!/usr/bin/env python3
"""Evaluate trained agent against baselines."""

from stable_baselines3 import PPO
from ..core.rl_wrapper import BigTwoRLWrapper
from ..agents import RandomAgent, GreedyAgent, PPOAgent, BaseAgent
from .tournament import Tournament


class Evaluator:
    """High-level evaluator for Big Two agents."""

    def __init__(self, num_games: int = 100):
        """
        Initialize evaluator.

        Args:
            num_games: Number of games per evaluation
        """
        self.num_games = num_games

    def evaluate_agent(self, agent: BaseAgent, baselines: bool = True) -> dict:
        """
        Evaluate an agent against baseline agents.

        Args:
            agent: Agent to evaluate
            baselines: Whether to include random/greedy baselines

        Returns:
            Dict with evaluation results
        """
        agents = [agent]

        if baselines:
            agents.extend([RandomAgent("Random"), GreedyAgent("Greedy")])

        tournament = Tournament(agents)
        return tournament.run_round_robin(self.num_games)

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

        agents = []
        for path, name in zip(model_paths, model_names):
            agents.append(PPOAgent(path, name))

        # Add baselines
        agents.extend([RandomAgent("Random"), GreedyAgent("Greedy")])

        tournament = Tournament(agents)
        return tournament.run_round_robin(self.num_games)


def evaluate_agent(model_path, num_games=100):
    """Evaluate agent against baselines."""

    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    env = BigTwoRLWrapper()
    random_agent = RandomAgent("Random")
    greedy_agent = GreedyAgent("Greedy")
    greedy_agent.set_env_reference(env)

    results = {"vs_random": 0, "vs_greedy": 0}

    # Test vs random policy
    print("Evaluating vs random policy...")
    wins = 0
    for game in range(num_games):
        obs, _ = env.reset()
        done = False

        while not done:
            if env.env.current_player == 0:  # Agent's turn
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            else:  # Random opponent
                action_mask = env.get_action_mask()
                action = random_agent.get_action(obs, action_mask)

            obs, reward, done, _, _ = env.step(action)

            if done and reward > 0:  # Agent won
                wins += 1
                break

    results["vs_random"] = wins / num_games
    print(f"Win rate vs random: {results['vs_random']:.2%}")

    # Test vs greedy policy
    print("Evaluating vs greedy policy...")
    wins = 0
    for game in range(num_games):
        obs, _ = env.reset()
        done = False

        while not done:
            if env.env.current_player == 0:  # Agent's turn
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            else:  # Greedy opponent
                action_mask = env.get_action_mask()
                action = greedy_agent.get_action(obs, action_mask)

            obs, reward, done, _, _ = env.step(action)

            if done and reward > 0:  # Agent won
                wins += 1
                break

    results["vs_greedy"] = wins / num_games
    print(f"Win rate vs greedy: {results['vs_greedy']:.2%}")

    return results


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "./models/best_model"
    results = evaluate_agent(model_path)
