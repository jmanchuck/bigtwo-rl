"""Tournament system for fixed 1,365-action space.

This module provides tournament and evaluation capabilities for agents
using the fixed action space, with proper action masking and compatibility.
"""

import numpy as np
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import pickle
import traceback

from .tournament import Tournament  # Import base tournament for inheritance
from ..agents.base_agent import BaseAgent
from ..agents.fixed_action_ppo_agent import FixedActionPPOAgent
from ..agents.fixed_action_random_agent import FixedActionRandomAgent
from ..agents.fixed_action_greedy_agent import FixedActionGreedyAgent
from ..core.fixed_action_wrapper import FixedActionBigTwoWrapper
from ..core.observation_builder import standard_observation


class FixedActionTournament(Tournament):
    """Tournament system using fixed 1,365-action space.
    
    This extends the base Tournament class to work with the fixed action space
    and provides proper agent validation and game execution.
    """
    
    def __init__(
        self, 
        agents: List[BaseAgent], 
        n_processes: Optional[int] = None,
        verbose: bool = True
    ):
        """Initialize Fixed Action Tournament.
        
        Args:
            agents: List of agents to compete (must support fixed actions)
            n_processes: Number of processes for parallel execution
            verbose: Whether to print progress information
        """
        # Initialize base tournament first
        super().__init__(agents, n_processes=n_processes)
        self.verbose = verbose
        
        # Validate all agents are compatible with fixed action space
        self._validate_agents(agents)
        
        if self.verbose:
            print(f"🏆 FixedActionTournament initialized with {len(agents)} agents")
            print(f"⚡ Parallel execution: {n_processes if n_processes else 'auto-detect'} processes")
    
    def _validate_agents(self, agents: List[BaseAgent]) -> None:
        """Validate that all agents work with fixed action space.
        
        Args:
            agents: List of agents to validate
        """
        for agent in agents:
            # Check if agent has appropriate get_action signature
            if not hasattr(agent, 'get_action'):
                raise ValueError(f"Agent {agent.name} must have get_action method")
            
            # For PPO agents, check model compatibility
            if isinstance(agent, FixedActionPPOAgent):
                if hasattr(agent, 'model') and agent.model is not None:
                    if agent.model.action_space.n != 1365:
                        raise ValueError(
                            f"PPO agent {agent.name} has wrong action space size: "
                            f"{agent.model.action_space.n} (expected 1365)"
                        )
            
            if self.verbose:
                print(f"✅ Validated agent: {agent.name}")
    
    def _create_game_environment(self) -> FixedActionBigTwoWrapper:
        """Create game environment for tournament."""
        return FixedActionBigTwoWrapper(
            observation_config=standard_observation(),
            games_per_episode=1,  # Single game per environment step
            reward_function=None,  # Tournament doesn't need training rewards
            track_move_history=False,
        )
    
    def play_game(self, agents: List[BaseAgent], game_id: Optional[int] = None) -> Dict[str, Any]:
        """Play a single 4-player game with fixed actions.
        
        Args:
            agents: List of exactly 4 agents
            game_id: Optional game ID for tracking
            
        Returns:
            Dictionary with game results
        """
        if len(agents) != 4:
            raise ValueError(f"Need exactly 4 agents, got {len(agents)}")
        
        try:
            env = self._create_game_environment()
            obs, info = env.reset()
            
            done = False
            step_count = 0
            max_steps = 1000  # Safety limit
            game_history = []
            
            while not done and step_count < max_steps:
                current_player = env.game.current_player
                agent = agents[current_player]
                
                # Get action mask for legal moves
                action_mask = env.get_action_mask()
                
                # Get agent action
                try:
                    action = agent.get_action(obs, action_mask=action_mask)
                    
                    # Validate action is legal
                    if not action_mask[action]:
                        if self.verbose:
                            print(f"⚠️  Agent {agent.name} selected illegal action {action}")
                        # Force a legal action
                        legal_actions = np.where(action_mask)[0]
                        if len(legal_actions) > 0:
                            action = legal_actions[0]
                        else:
                            raise ValueError("No legal actions available")
                    
                    # Record move for history
                    game_history.append({
                        'step': step_count,
                        'player': current_player,
                        'agent': agent.name,
                        'action': action,
                        'legal_actions_count': np.sum(action_mask)
                    })
                    
                except Exception as e:
                    if self.verbose:
                        print(f"❌ Agent {agent.name} error: {e}")
                    # Force a legal action as fallback
                    legal_actions = np.where(action_mask)[0]
                    action = legal_actions[0] if len(legal_actions) > 0 else 0
                
                # Execute action
                obs, reward, done, truncated, info = env.step(action)
                step_count += 1
            
            # Get final results
            winner = None
            final_scores = [0] * 4
            
            if done and hasattr(env.game, 'hands'):
                # Calculate final scores (cards remaining)
                final_scores = [int(np.sum(env.game.hands[i])) for i in range(4)]
                
                # Find winner (player with 0 cards or lowest score)
                min_score = min(final_scores)
                if min_score == 0:
                    winner = final_scores.index(0)
                else:
                    winner = final_scores.index(min_score)
            
            return {
                'game_id': game_id,
                'winner': winner,
                'final_scores': final_scores,
                'steps': step_count,
                'completed': done,
                'truncated': truncated,
                'max_steps_reached': step_count >= max_steps,
                'game_history': game_history if self.verbose else [],
                'agents': [agent.name for agent in agents]
            }
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Game failed: {e}")
                traceback.print_exc()
            
            return {
                'game_id': game_id,
                'winner': None,
                'final_scores': [13, 13, 13, 13],  # All players keep all cards
                'steps': 0,
                'completed': False,
                'error': str(e),
                'agents': [agent.name for agent in agents]
            }
    
    def run(self, num_games: int, **kwargs) -> Dict[str, Any]:
        """Run tournament with fixed action space.
        
        Args:
            num_games: Number of games to play
            **kwargs: Additional arguments
            
        Returns:
            Tournament results
        """
        if self.verbose:
            print(f"🎮 Starting tournament with {num_games} games")
        
        start_time = time.time()
        
        # Run tournament using base class logic but with fixed action validation
        results = super().run(num_games, **kwargs)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if self.verbose:
            print(f"🏁 Tournament completed in {duration:.2f}s")
            print(f"📊 Games per second: {num_games / duration:.2f}")
        
        # Add fixed action space metadata
        results['tournament_type'] = 'FixedActionTournament'
        results['action_space_size'] = 1365
        results['uses_action_masking'] = True
        results['duration_seconds'] = duration
        
        return results


class FixedActionSeriesEvaluator:
    """Evaluator for running series of games between specific agents."""
    
    def __init__(self, verbose: bool = True):
        """Initialize series evaluator.
        
        Args:
            verbose: Whether to print progress information
        """
        self.verbose = verbose
    
    def play_four_player_series(
        self,
        agents: List[BaseAgent],
        num_games: int = 100,
        n_processes: Optional[int] = None
    ) -> Dict[str, Any]:
        """Play series of 4-player games with fixed action space.
        
        Args:
            agents: List of exactly 4 agents
            num_games: Number of games to play
            n_processes: Number of parallel processes
            
        Returns:
            Series results with detailed statistics
        """
        if len(agents) != 4:
            raise ValueError(f"Need exactly 4 agents, got {len(agents)}")
        
        if self.verbose:
            agent_names = [agent.name for agent in agents]
            print(f"🎯 Starting 4-player series: {' vs '.join(agent_names)}")
            print(f"🎮 Games: {num_games}, Processes: {n_processes or 'auto'}")
        
        tournament = FixedActionTournament(agents, n_processes=n_processes, verbose=False)
        
        start_time = time.time()
        results = []
        
        if n_processes and n_processes > 1:
            # Parallel execution
            results = self._run_parallel_series(tournament, agents, num_games, n_processes)
        else:
            # Sequential execution
            for game_id in range(num_games):
                if self.verbose and (game_id + 1) % 20 == 0:
                    print(f"📈 Progress: {game_id + 1}/{num_games} games")
                
                game_result = tournament.play_game(agents, game_id)
                results.append(game_result)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Aggregate results
        return self._aggregate_series_results(results, agents, num_games, duration)
    
    def _run_parallel_series(
        self,
        tournament: FixedActionTournament,
        agents: List[BaseAgent],
        num_games: int,
        n_processes: int
    ) -> List[Dict[str, Any]]:
        """Run series using parallel processing.
        
        Args:
            tournament: Tournament instance
            agents: List of agents
            num_games: Number of games
            n_processes: Number of processes
            
        Returns:
            List of game results
        """
        # Create serializable agent data for multiprocessing
        agent_data = []
        for agent in agents:
            if isinstance(agent, FixedActionPPOAgent):
                agent_data.append({
                    'type': 'ppo',
                    'name': agent.name,
                    'model_path': agent.model_path,
                    'deterministic': agent.deterministic
                })
            elif isinstance(agent, FixedActionRandomAgent):
                agent_data.append({
                    'type': 'random',
                    'name': agent.name,
                    'seed': getattr(agent, 'seed', None)
                })
            elif isinstance(agent, FixedActionGreedyAgent):
                agent_data.append({
                    'type': 'greedy',
                    'name': agent.name,
                    'strategy': agent.strategy
                })
            else:
                raise ValueError(f"Unsupported agent type for parallel execution: {type(agent)}")
        
        # Run games in parallel
        batch_size = max(1, num_games // n_processes)
        results = []
        
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            futures = []
            
            for i in range(0, num_games, batch_size):
                batch_games = min(batch_size, num_games - i)
                future = executor.submit(
                    _play_game_batch,
                    agent_data,
                    batch_games,
                    i  # start_game_id
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    
                    if self.verbose:
                        print(f"📈 Completed batch: {len(results)}/{num_games} games")
                        
                except Exception as e:
                    print(f"❌ Batch failed: {e}")
        
        return results
    
    def _aggregate_series_results(
        self,
        results: List[Dict[str, Any]],
        agents: List[BaseAgent],
        num_games: int,
        duration: float
    ) -> Dict[str, Any]:
        """Aggregate series results into summary statistics.
        
        Args:
            results: List of game results
            agents: List of agents
            num_games: Total games played
            duration: Total duration
            
        Returns:
            Aggregated results
        """
        # Initialize counters
        wins = [0] * 4
        total_scores = [0] * 4
        completed_games = 0
        
        for result in results:
            if result['completed'] and result['winner'] is not None:
                wins[result['winner']] += 1
                completed_games += 1
            
            for i, score in enumerate(result['final_scores']):
                total_scores[i] += score
        
        # Calculate statistics
        win_rates = [w / completed_games if completed_games > 0 else 0 for w in wins]
        avg_scores = [s / len(results) if results else 0 for s in total_scores]
        
        aggregated_results = {
            'agents': [agent.name for agent in agents],
            'total_games': num_games,
            'completed_games': completed_games,
            'wins': wins,
            'win_rates': win_rates,
            'avg_scores': avg_scores,
            'duration_seconds': duration,
            'games_per_second': num_games / duration if duration > 0 else 0,
            'game_results': results,
            'series_type': 'four_player_fixed_action'
        }
        
        if self.verbose:
            print(f"\n🏆 Series Results:")
            for i, agent in enumerate(agents):
                print(f"  {agent.name}: {wins[i]} wins ({win_rates[i]:.1%}), avg cards: {avg_scores[i]:.1f}")
        
        return aggregated_results


def _play_game_batch(agent_data: List[Dict], num_games: int, start_game_id: int) -> List[Dict[str, Any]]:
    """Play a batch of games (for multiprocessing).
    
    Args:
        agent_data: Serializable agent data
        num_games: Number of games in batch
        start_game_id: Starting game ID
        
    Returns:
        List of game results
    """
    try:
        # Recreate agents from data
        agents = []
        for data in agent_data:
            if data['type'] == 'ppo':
                agent = FixedActionPPOAgent(
                    model_path=data['model_path'],
                    name=data['name'],
                    deterministic=data['deterministic']
                )
            elif data['type'] == 'random':
                agent = FixedActionRandomAgent(
                    name=data['name'],
                    seed=data['seed']
                )
            elif data['type'] == 'greedy':
                agent = FixedActionGreedyAgent(
                    name=data['name'],
                    strategy=data['strategy']
                )
            else:
                raise ValueError(f"Unknown agent type: {data['type']}")
            agents.append(agent)
        
        # Create tournament and play games
        tournament = FixedActionTournament(agents, verbose=False)
        results = []
        
        for i in range(num_games):
            game_id = start_game_id + i
            result = tournament.play_game(agents, game_id)
            results.append(result)
        
        return results
        
    except Exception as e:
        # Return error results for failed batch
        error_results = []
        for i in range(num_games):
            error_results.append({
                'game_id': start_game_id + i,
                'winner': None,
                'final_scores': [13, 13, 13, 13],
                'steps': 0,
                'completed': False,
                'error': f"Batch processing error: {e}",
                'agents': [data['name'] for data in agent_data]
            })
        return error_results


# Convenience function for backward compatibility
def play_four_player_series_fixed(
    agents: List[BaseAgent],
    num_games: int = 100,
    n_processes: Optional[int] = None
) -> Dict[str, Any]:
    """Play series of games with fixed action space (convenience function).
    
    Args:
        agents: List of exactly 4 agents
        num_games: Number of games to play
        n_processes: Number of processes for parallel execution
        
    Returns:
        Series results
    """
    evaluator = FixedActionSeriesEvaluator(verbose=True)
    return evaluator.play_four_player_series(agents, num_games, n_processes)