#!/usr/bin/env python3
"""Tournament system for Big Two agents."""

import numpy as np
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional
from ..core.rl_wrapper import BigTwoRLWrapper
from ..core.observation_builder import ObservationConfig
from ..agents import BaseAgent, RandomAgent, GreedyAgent, PPOAgent


class Tournament:
    """High-level tournament system for Big Two agents."""

    def __init__(self, agents: List[BaseAgent], n_processes: Optional[int] = None):
        """
        Initialize tournament with list of agents.

        Args:
            agents: List of BaseAgent instances to compete
            n_processes: Number of processes for parallel execution (None = auto)
        """
        self.agents = agents
        self.n_processes = n_processes

    def run(self, num_games: int = 100) -> Dict:
        """
        Run round-robin tournament where each agent plays against every other agent.

        Args:
            num_games: Number of games per matchup

        Returns:
            Dict with tournament results and statistics
        """
        return run_tournament(self.agents, num_games, self.n_processes)

    def run_parallel(
        self, num_games: int = 100, n_processes: Optional[int] = None
    ) -> Dict:
        """
        Run tournament with explicit parallel processing control.

        Args:
            num_games: Number of games per matchup
            n_processes: Override default process count

        Returns:
            Dict with tournament results and statistics
        """
        processes = n_processes or self.n_processes
        return run_tournament(self.agents, num_games, processes)

    def add_agent(self, agent: BaseAgent) -> None:
        """Add an agent to the tournament."""
        self.agents.append(agent)

    def remove_agent(self, agent_name: str) -> None:
        """Remove an agent by name."""
        self.agents = [a for a in self.agents if a.name != agent_name]

    def list_agents(self) -> List[str]:
        """Get list of agent names."""
        return [agent.name for agent in self.agents]


def play_single_game(
    agents: List[BaseAgent], env: BigTwoRLWrapper
) -> Tuple[int, float, List[int]]:
    """Play a single game between multiple agents.

    Returns:
        (winner_idx, reward, cards_remaining):
            winner_idx: index of winner in agents list, or -1 if none
            reward: terminal reward returned by env for current player
            cards_remaining: list of number of cards remaining per player at game end
    """
    obs, _ = env.reset()
    done = False

    # Reset all agents and provide env reference for observation mapping
    for agent in agents:
        agent.reset()
        if hasattr(agent, "set_env_reference"):
            agent.set_env_reference(env)  # ty: ignore

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

        if done:
            # Game finished - determine winner directly from hands, not from reward
            winner_idx = -1
            for p in range(env.env.num_players):
                if np.sum(env.env.hands[p]) == 0:
                    winner_idx = p
                    break

            # Record results for all agents
            for i, agent in enumerate(agents):
                if i < env.env.num_players:  # Only count agents that actually played
                    agent.record_game_result(i == winner_idx)

            # Capture cards remaining for each player
            cards_remaining = [
                np.sum(env.env.hands[p]) for p in range(env.env.num_players)
            ]
            return winner_idx, reward, cards_remaining

    # No winner (should not happen); capture current hand sizes
    cards_remaining = [np.sum(env.env.hands[p]) for p in range(env.env.num_players)]
    return -1, 0, cards_remaining


def _serialize_agent(agent: BaseAgent) -> Dict:
    """Serialize agent for multiprocessing."""
    if isinstance(agent, RandomAgent):
        return {"type": "random", "name": agent.name}
    elif isinstance(agent, GreedyAgent):
        return {"type": "greedy", "name": agent.name}
    elif isinstance(agent, PPOAgent):
        if agent.model_path is None:
            raise ValueError(
                f"Cannot serialize PPOAgent '{agent.name}' without model_path"
            )
        return {"type": "ppo", "name": agent.name, "model_path": agent.model_path}
    else:
        raise ValueError(f"Cannot serialize agent type: {type(agent)}")


def _deserialize_agent(agent_config: Dict) -> BaseAgent:
    """Deserialize agent from config."""
    agent_type = agent_config["type"]
    name = agent_config["name"]

    if agent_type == "random":
        return RandomAgent(name)
    elif agent_type == "greedy":
        return GreedyAgent(name)
    elif agent_type == "ppo":
        return PPOAgent(model_path=agent_config["model_path"], name=name)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def _union_observation_config_for_agents(agents: List[BaseAgent]) -> ObservationConfig:
    """Create an observation configuration that contains the union of features
    required by the provided agents' models.

    - If agents are PPO and have model metadata, use that to determine their
      model feature needs. If not available, fall back to standard observation.
    - The union config ensures the environment can generate all features any
      agent might require. Agents that do not use certain features will ignore
      them via their observation converter.
    """
    # Start with standard config to maintain backward compatibility
    from ..core.observation_builder import standard_observation

    union = standard_observation()

    def widen(to_union: ObservationConfig, other: ObservationConfig) -> None:
        # Flip on any feature that the other config uses
        for attr in [
            "include_hand",
            "include_last_play",
            "include_hand_sizes",
            "include_played_cards",
            "include_remaining_deck",
            "include_cards_by_player",
            "include_last_play_exists",
            "include_game_phase",
            "include_turn_position",
            "include_trick_history",
            "include_pass_history",
            "include_play_patterns",
            "include_power_cards_remaining",
            "include_hand_type_capabilities",
        ]:
            if getattr(other, attr):
                setattr(to_union, attr, True)

    for agent in agents:
        # Prefer explicit exposure of model configuration if present
        model_cfg = getattr(agent, "model_obs_config", None)
        if isinstance(model_cfg, ObservationConfig):
            widen(union, model_cfg)
        else:
            # As a fallback, if the agent was initialized with env_obs_config
            env_cfg = getattr(agent, "env_obs_config", None)
            if isinstance(env_cfg, ObservationConfig):
                widen(union, env_cfg)

    # Recompute sizes
    union.__post_init__()
    return union


def _run_game_batch(
    args: Tuple[List[Dict], int, int],
) -> Tuple[Dict[str, int], Dict[str, int], List[List[int]]]:
    """
    Worker function to run a batch of games in parallel.

    Args:
        args: Tuple of (agent_configs, num_games, random_seed)

    Returns:
        Tuple of (wins_dict, cards_left_sum_dict, cards_history)
    """
    agent_configs, num_games, random_seed = args

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Deserialize agents
    agents = [_deserialize_agent(config) for config in agent_configs]

    # Create environment with observation space that supports all agents
    env_obs_config = _union_observation_config_for_agents(agents)
    env = BigTwoRLWrapper(
        num_players=4, games_per_episode=1, observation_config=env_obs_config
    )

    # Initialize tracking
    wins = {a.name: 0 for a in agents}
    cards_left_sum = {a.name: 0 for a in agents}
    cards_history = []

    # Run games
    for _ in range(num_games):
        winner_idx, _, cards_remaining = play_single_game(agents, env)
        if winner_idx != -1:
            wins[agents[winner_idx].name] += 1
        # Note: winner_idx should never be -1 in Big Two, but keeping check for safety

        # Accumulate cards remaining
        for i, agent in enumerate(agents):
            cards_left_sum[agent.name] += cards_remaining[i]
        cards_history.append(cards_remaining)

    return wins, cards_left_sum, cards_history


def play_four_player_series(
    agents: List[BaseAgent], num_games: int = 100, n_processes: Optional[int] = None
) -> Dict:
    """Play a series of 4-player matches between the provided four agents.

    Args:
        agents: Exactly four agents who will play together
        num_games: Number of games to play
        n_processes: Number of processes to use (None = auto-detect, 1 = sequential)

    Returns:
        Dict with per-agent wins and win rates for the series
    """
    if len(agents) != 4:
        raise ValueError("play_four_player_series requires exactly 4 agents")

    # Use sequential version if n_processes=1 or num_games is small
    if n_processes == 1 or num_games < 10:
        return _play_four_player_series_sequential(agents, num_games)

    # Determine number of processes
    if n_processes is None:
        n_processes = min(mp.cpu_count(), num_games)

    # Serialize agents for multiprocessing
    agent_configs = [_serialize_agent(agent) for agent in agents]

    # Split games across processes
    games_per_process = num_games // n_processes
    remaining_games = num_games % n_processes

    # Create work batches
    work_batches = []
    for i in range(n_processes):
        batch_games = games_per_process + (1 if i < remaining_games else 0)
        if batch_games > 0:
            # Use different random seeds for each process
            random_seed = np.random.randint(0, 2**31 - 1)
            work_batches.append((agent_configs, batch_games, random_seed))

    # Run batches in parallel
    with mp.Pool(processes=len(work_batches)) as pool:
        results = pool.map(_run_game_batch, work_batches)

    # Aggregate results
    total_wins = {a.name: 0 for a in agents}
    total_cards_left_sum = {a.name: 0 for a in agents}
    all_cards_history = []

    for wins, cards_left_sum, cards_history in results:
        for name in total_wins:
            total_wins[name] += wins[name]
            total_cards_left_sum[name] += cards_left_sum[name]
        all_cards_history.extend(cards_history)

    # Update agent stats (approximate since we can't update across processes)
    for i, agent in enumerate(agents):
        agent.wins += total_wins[agent.name]
        agent.games_played += num_games

    return {
        "players": [a.name for a in agents],
        "wins": total_wins,
        "win_rates": {name: wins / num_games for name, wins in total_wins.items()},
        "avg_cards_left": {
            name: total_cards_left_sum[name] / num_games
            for name in total_cards_left_sum
        },
        "cards_left_by_game": all_cards_history,
        "games_played": num_games,
    }


def _play_four_player_series_sequential(
    agents: List[BaseAgent], num_games: int
) -> Dict:
    """Sequential version of play_four_player_series for comparison and small runs."""
    env_obs_config = _union_observation_config_for_agents(agents)
    env = BigTwoRLWrapper(
        num_players=4, games_per_episode=1, observation_config=env_obs_config
    )

    # Local aggregates per agent
    local_wins = {a.name: 0 for a in agents}
    local_cards_left_sum = {a.name: 0 for a in agents}
    cards_left_history: List[List[int]] = []

    for _ in range(num_games):
        winner_idx, _, cards_remaining = play_single_game(agents, env)
        if winner_idx != -1:
            local_wins[agents[winner_idx].name] += 1
        # Note: winner_idx should never be -1 in Big Two, but keeping check for safety

        # Accumulate cards remaining per player
        for i, a in enumerate(agents):
            local_cards_left_sum[a.name] += cards_remaining[i]
        cards_left_history.append(cards_remaining)

    return {
        "players": [a.name for a in agents],
        "wins": local_wins,
        "win_rates": {name: wins / num_games for name, wins in local_wins.items()},
        "avg_cards_left": {
            name: local_cards_left_sum[name] / num_games
            for name in local_cards_left_sum
        },
        "cards_left_by_game": cards_left_history,
        "games_played": num_games,
    }


def run_tournament(
    agents: List[BaseAgent], num_games: int = 100, n_processes: Optional[int] = None
) -> Dict:
    """Run a round-robin tournament over all 4-player combinations of the agents."""
    if len(agents) != 4:
        raise ValueError("Tournament requires exactly 4 agents")

    matchup_results = []

    # Reset all agent stats
    for agent in agents:
        agent.reset_stats()

    parallel_info = ""
    if n_processes != 1 and num_games >= 10:
        processes = n_processes or min(mp.cpu_count(), num_games)
        parallel_info = f" ({processes} processes)"

    print(
        f"Running 4-player tournament with {len(agents)} agents, {num_games} games{parallel_info}..."
    )

    result = play_four_player_series(agents, num_games, n_processes)
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


def _create_tournament_summary(
    agents: List[BaseAgent], matchup_results: List[Dict]
) -> str:
    """Create a readable tournament summary for 4-player tables."""
    summary = []
    summary.append("Tournament Results:")
    summary.append("=" * 50)

    # Sort agents by win rate
    # Use agent index to disambiguate duplicate names in the summary output
    indexed_agents = list(enumerate(agents))  # list of (index, agent)
    sorted_indexed_agents = sorted(
        indexed_agents, key=lambda ia: ia[1].get_win_rate(), reverse=True
    )

    for rank, (orig_index, agent) in enumerate(sorted_indexed_agents, start=1):
        display_name = f"{agent.name}#{orig_index+1}"
        summary.append(
            f"{rank}. {display_name}: {agent.wins}/{agent.games_played} wins ({agent.get_win_rate():.2%})"
        )

    # Add card statistics from the matchup results
    if matchup_results:
        result = matchup_results[0]  # Single 4-player table result
        summary.append("\nCard Statistics:")
        summary.append("-" * 30)

        # Calculate total cards left for each player across all games
        total_cards_by_player = {}
        for name in result["players"]:
            total_cards_by_player[name] = (
                result["avg_cards_left"][name] * result["games_played"]
            )

        # Sort by average cards left (ascending - fewer cards left is better)
        sorted_by_avg_cards = sorted(
            result["avg_cards_left"].items(), key=lambda x: x[1]
        )

        for name, avg_cards in sorted_by_avg_cards:
            total_cards = int(total_cards_by_player[name])
            summary.append(
                f"{name}: {total_cards} total cards left, {avg_cards:.1f} avg cards left"
            )

    summary.append("\nTable Results:")
    summary.append("-" * 30)
    for result in matchup_results:
        players = ", ".join(result["players"])  # type: ignore[index]
        wins_str = ", ".join(
            f"{name}: {result['wins'][name]}" for name in result["players"]
        )  # type: ignore[index]
        summary.append(f"{players} | {wins_str}")

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
        results = run_tournament(agents, num_games=50)
        print(results["tournament_summary"])
    else:
        print("Usage: python tournament.py example")
        print("Or import and use the functions directly")
