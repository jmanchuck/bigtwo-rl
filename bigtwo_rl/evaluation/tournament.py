#!/usr/bin/env python3
"""Tournament system for Big Two agents."""

import numpy as np
from typing import List, Dict
import itertools
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
        Head-to-head is not supported. Big Two is a 4-player game.
        Use run_round_robin with at least 4 agents instead.
        """
        raise NotImplementedError("Head-to-head is not supported; use run_round_robin with at least 4 agents.")

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
    """Play a single game between multiple agents.

    Returns:
        (winner_idx, reward, cards_remaining):
            winner_idx: index of winner in agents list, or -1 if none
            reward: terminal reward returned by env for current player
            cards_remaining: list of number of cards remaining per player at game end
    """
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
                if np.sum(env.env.hands[p]) == 0:
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

            # Capture cards remaining for each player
            cards_remaining = [np.sum(env.env.hands[p]) for p in range(env.env.num_players)]
            return winner_idx, reward, cards_remaining

    # No winner (should not happen); capture current hand sizes
    cards_remaining = [np.sum(env.env.hands[p]) for p in range(env.env.num_players)]
    return -1, 0, cards_remaining


def play_four_player_series(agents: List[BaseAgent], num_games: int = 100) -> Dict:
    """Play a series of 4-player matches between the provided four agents.

    Args:
        agents: Exactly four agents who will play together
        num_games: Number of games to play

    Returns:
        Dict with per-agent wins and win rates for the series
    """
    if len(agents) != 4:
        raise ValueError("play_four_player_series requires exactly 4 agents")

    env = BigTwoRLWrapper(num_players=4, games_per_episode=1)

    # Local aggregates per agent
    local_wins = {a.name: 0 for a in agents}
    local_cards_left_sum = {a.name: 0 for a in agents}
    draws = 0
    cards_left_history: List[List[int]] = []

    for _ in range(num_games):
        winner_idx, _, cards_remaining = play_single_game(agents, env)
        if winner_idx == -1:
            draws += 1
        else:
            local_wins[agents[winner_idx].name] += 1

        # Accumulate cards remaining per player
        for i, a in enumerate(agents):
            local_cards_left_sum[a.name] += cards_remaining[i]
        cards_left_history.append(cards_remaining)

    total_games = num_games
    return {
        "players": [a.name for a in agents],
        "wins": local_wins,
        "win_rates": {name: wins / total_games for name, wins in local_wins.items()},
        "avg_cards_left": {name: local_cards_left_sum[name] / total_games for name in local_cards_left_sum},
        "draws": draws,
        "cards_left_by_game": cards_left_history,
        "games_played": total_games,
    }


def run_round_robin_tournament(agents: List[BaseAgent], num_games: int = 100) -> Dict:
    """Run a round-robin tournament over all 4-player combinations of the agents."""
    if len(agents) < 4:
        raise ValueError("Tournament requires at least 4 agents")

    matchup_results = []

    # Reset all agent stats
    for agent in agents:
        agent.reset_stats()

    print(f"Running 4-player round-robin tournament with {len(agents)} agents, {num_games} games per table...")

    # Play all 4-agent combinations
    for combo in itertools.combinations(range(len(agents)), 4):
        group_agents = [agents[i] for i in combo]
        group_names = ", ".join(a.name for a in group_agents)
        print(f"Playing table: {group_names}...")

        result = play_four_player_series(group_agents, num_games)
        matchup_results.append(result)

    # Calculate overall stats
    agent_stats: Dict[str, Dict] = {}
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
    """Create a readable tournament summary for 4-player tables."""
    summary = []
    summary.append("Tournament Results:")
    summary.append("=" * 50)

    # Sort agents by win rate
    sorted_agents = sorted(agents, key=lambda a: a.get_win_rate(), reverse=True)

    for i, agent in enumerate(sorted_agents):
        summary.append(f"{i+1}. {agent.name}: {agent.wins}/{agent.games_played} wins ({agent.get_win_rate():.2%})")

    summary.append("\nTable Results:")
    summary.append("-" * 30)
    for result in matchup_results:
        players = ", ".join(result["players"])  # type: ignore[index]
        wins_str = ", ".join(f"{name}: {result['wins'][name]}" for name in result["players"])  # type: ignore[index]
        summary.append(f"{players} | {wins_str} | draws: {result['draws']}")

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
        except FileNotFoundError:
            print("No trained PPO model found, skipping PPO agent")
        except Exception as e:
            print(f"Failed to load PPO agent: {e}")

        # Run tournament
        results = run_round_robin_tournament(agents, num_games=50)
        print(results["tournament_summary"])
    else:
        print("Usage: python tournament.py example")
        print("Or import and use the functions directly")
