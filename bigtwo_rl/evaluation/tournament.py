#!/usr/bin/env python3
"""Tournament system for Big Two agents."""

import numpy as np
from typing import List, Dict
from ..core.rl_wrapper import BigTwoRLWrapper
from ..agents import BaseAgent, RandomAgent, GreedyAgent, PPOAgent


class Tournament:
    """High-level tournament system for Big Two agents."""

    def __init__(self, agents: List[BaseAgent]):
        """
        Initialize tournament with list of agents.

        Args:
            agents: List of BaseAgent instances to compete
        """
        self.agents = agents

    def run_round_robin(self, num_games: int = 100) -> Dict:
        """
        Run round-robin tournament where each agent plays against every other agent.

        Args:
            num_games: Number of games per matchup

        Returns:
            Dict with tournament results and statistics
        """
        return run_round_robin_tournament(self.agents, num_games)

    def head_to_head(self, agent1_idx: int = 0, agent2_idx: int = 1, num_games: int = 100) -> Dict:
        """
        Run head-to-head match between two agents.

        Args:
            agent1_idx: Index of first agent
            agent2_idx: Index of second agent
            num_games: Number of games to play

        Returns:
            Dict with head-to-head results
        """
        if agent1_idx >= len(self.agents) or agent2_idx >= len(self.agents):
            raise ValueError("Agent index out of range")

        return play_head_to_head(self.agents[agent1_idx], self.agents[agent2_idx], num_games)

    def add_agent(self, agent: BaseAgent):
        """Add an agent to the tournament."""
        self.agents.append(agent)

    def remove_agent(self, agent_name: str):
        """Remove an agent by name."""
        self.agents = [a for a in self.agents if a.name != agent_name]

    def list_agents(self) -> List[str]:
        """Get list of agent names."""
        return [agent.name for agent in self.agents]


def play_single_game(agents: List[BaseAgent], env: BigTwoRLWrapper):
    """Play a single game between multiple agents."""
    obs, _ = env.reset()
    done = False

    # Reset all agents
    for agent in agents:
        agent.reset()
        if hasattr(agent, "set_env_reference"):
            agent.set_env_reference(env)

    last_winner = -1

    while not done:
        current_player = env.env.current_player
        if current_player < len(agents):
            agent = agents[current_player]
            action_mask = env.get_action_mask()

            action = agent.get_action(obs, action_mask)
        else:
            # Random player if not enough agents
            action_mask = env.get_action_mask()
            legal_actions = np.where(action_mask)[0]
            action = np.random.choice(legal_actions) if len(legal_actions) > 0 else 0

        obs, reward, done, _, info = env.step(action)

        # Track who won based on the environment's internal state
        if hasattr(env.env, "done") and env.env.done:
            # Find winner (player with 0 cards)
            for p in range(env.env.num_players):
                if len(env.env.hands[p]) == 0:
                    last_winner = p
                    break

        if done:
            # Game finished - use tracked winner or reward signal
            winner_idx = last_winner if last_winner != -1 else (0 if reward > 0 else -1)

            # Record results for all agents
            for i, agent in enumerate(agents):
                if i < env.env.num_players:  # Only count agents that actually played
                    won = i == winner_idx
                    agent.record_game_result(won)

            return winner_idx, reward

    return -1, 0  # No winner


def play_head_to_head(agent1: BaseAgent, agent2: BaseAgent, num_games: int = 100) -> Dict:
    """Play head-to-head matches between two agents."""
    env = BigTwoRLWrapper(num_players=2, games_per_episode=1)

    # Reset stats
    agent1.reset_stats()
    agent2.reset_stats()

    results = {"agent1_wins": 0, "agent2_wins": 0, "draws": 0}

    for game in range(num_games):
        agents = [agent1, agent2]
        winner_idx, _ = play_single_game(agents, env)

        if winner_idx == 0:
            results["agent1_wins"] += 1
        elif winner_idx == 1:
            results["agent2_wins"] += 1
        else:
            results["draws"] += 1

    return {
        "agent1": agent1.name,
        "agent2": agent2.name,
        "agent1_wins": results["agent1_wins"],
        "agent2_wins": results["agent2_wins"],
        "agent1_win_rate": results["agent1_wins"] / num_games,
        "agent2_win_rate": results["agent2_wins"] / num_games,
        "games_played": num_games,
    }


def run_round_robin_tournament(agents: List[BaseAgent], num_games: int = 100) -> Dict:
    """Run round-robin tournament where each agent plays against every other agent."""
    matchup_results = []

    # Reset all agent stats
    for agent in agents:
        agent.reset_stats()

    print(f"Running round-robin tournament with {len(agents)} agents, {num_games} games per matchup...")

    # Play all pairs
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            agent1 = agents[i]
            agent2 = agents[j]

            print(f"Playing: {agent1.name} vs {agent2.name}...")
            matchup_result = play_head_to_head(agent1, agent2, num_games)
            matchup_results.append(matchup_result)

    # Calculate overall stats
    agent_stats = {}
    for agent in agents:
        agent_stats[agent.name] = {
            "total_wins": agent.wins,
            "total_games": agent.games_played,
            "win_rate": agent.get_win_rate(),
        }

    return {
        "agent_stats": agent_stats,
        "matchup_results": matchup_results,
        "tournament_summary": _create_tournament_summary(agents, matchup_results),
    }


def _create_tournament_summary(agents: List[BaseAgent], matchup_results: List[Dict]) -> str:
    """Create a readable tournament summary."""
    summary = []
    summary.append("Tournament Results:")
    summary.append("=" * 50)

    # Sort agents by win rate
    sorted_agents = sorted(agents, key=lambda a: a.get_win_rate(), reverse=True)

    for i, agent in enumerate(sorted_agents):
        summary.append(f"{i+1}. {agent.name}: {agent.wins}/{agent.games_played} wins ({agent.get_win_rate():.2%})")

    summary.append("\nHead-to-Head Results:")
    summary.append("-" * 30)
    for result in matchup_results:
        summary.append(
            f"{result['agent1']} vs {result['agent2']}: "
            f"{result['agent1_wins']}-{result['agent2_wins']} "
            f"({result['agent1_win_rate']:.2%} vs {result['agent2_win_rate']:.2%})"
        )

    return "\n".join(summary)


def load_agents_from_config(agent_configs: List[Dict]) -> List[BaseAgent]:
    """Load agents from configuration list."""
    agents = []

    for config in agent_configs:
        agent_type = config["type"]
        name = config.get("name", agent_type)

        if agent_type == "random":
            agents.append(RandomAgent(name))
        elif agent_type == "greedy":
            agents.append(GreedyAgent(name))
        elif agent_type == "ppo":
            model_path = config["model_path"]
            agents.append(PPOAgent(model_path=model_path, name=name))
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    return agents


if __name__ == "__main__":
    import sys

    # Example tournament
    if len(sys.argv) > 1 and sys.argv[1] == "example":
        # Create some example agents
        agents = [
            RandomAgent("Random"),
            GreedyAgent("Greedy"),
        ]

        # Add PPO agent if model exists
        try:
            agents.append(PPOAgent(model_path="./models/best_model", name="PPO-Best"))
        except:
            print("No trained PPO model found, skipping PPO agent")

        # Run tournament
        results = run_round_robin_tournament(agents, num_games=50)
        print(results["tournament_summary"])
    else:
        print("Usage: python tournament.py example")
        print("Or import and use the functions directly")
