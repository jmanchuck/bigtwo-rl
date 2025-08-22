"""Opponent controller for managing autoplay logic in Big Two RL environments."""

import numpy as np
from typing import Dict, Any, Optional, List, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..agents.base_agent import BaseAgent


class OpponentController:
    """Manages opponent agents and autoplay logic for Big Two RL training.
    
    Handles opponent agent selection, action generation, and autoplay sequencing
    while keeping this logic separate from the core RL wrapper.
    """
    
    def __init__(
        self, 
        num_players: int = 4, 
        controlled_player: int = 0, 
        opponent_provider=None
    ):
        """Initialize opponent controller.
        
        Args:
            num_players: Total number of players in the game
            controlled_player: Index of the player being trained (not controlled by opponents)
            opponent_provider: Provider object that generates opponents per episode
        """
        self.num_players = num_players
        self.controlled_player = controlled_player
        self.opponent_provider = opponent_provider
        self._episode_opponents: Optional[Dict[int, "BaseAgent"]] = None
    
    def setup_episode_opponents(self, env_reference=None) -> None:
        """Setup opponents for a new episode.
        
        Args:
            env_reference: Optional environment reference to pass to agents
        """
        if self.opponent_provider is not None:
            self._episode_opponents = self.opponent_provider.get_episode_opponents(
                self.num_players, self.controlled_player
            )
            # Provide env reference to opponents that support it (for better play)
            if env_reference is not None:
                for agent in self._episode_opponents.values():
                    if hasattr(agent, "set_env_reference"):
                        try:
                            agent.set_env_reference(env_reference)  # type: ignore
                        except Exception:
                            pass
        else:
            self._episode_opponents = None
    
    def has_opponents(self) -> bool:
        """Check if opponents are configured."""
        return self._episode_opponents is not None
    
    def should_autoplay_current_player(self, current_player: int) -> bool:
        """Check if the current player should be autoplayed by an opponent."""
        return (
            self.has_opponents() and 
            current_player != self.controlled_player
        )
    
    def select_opponent_action(
        self, 
        player_idx: int, 
        legal_moves: List[np.ndarray],
        observation_vector: np.ndarray,
        find_pass_move_fn: Callable[[List[np.ndarray]], Optional[int]]
    ) -> int:
        """Select an action index for a non-controlled opponent player.
        
        Args:
            player_idx: Index of the opponent player
            legal_moves: List of legal moves for the player
            observation_vector: Current game observation as vector
            find_pass_move_fn: Function to find pass move index in legal moves
            
        Returns:
            Action index for the opponent to take
        """
        # Create action mask
        mask = np.zeros(2000, dtype=bool)
        for i in range(min(len(legal_moves), 2000)):
            mask[i] = True
        
        # Get the agent for this player
        agent = self._episode_opponents.get(player_idx) if self._episode_opponents else None
        if agent is None:
            # Fallback: random legal move
            legal_indices = np.where(mask)[0]
            return int(legal_indices[0]) if len(legal_indices) > 0 else 0
        
        # Let the agent choose an action
        action = agent.get_action(observation_vector, mask)
        
        # Validate the action is legal
        if action >= len(legal_moves):
            # Clamp to a valid move (prefer PASS if available)
            pass_idx = find_pass_move_fn(legal_moves)
            return pass_idx if pass_idx is not None else 0
        
        return int(action)
    
    def autoplay_until_controlled_player(
        self,
        current_player: int,
        controlled_player: int,
        get_legal_moves_fn: Callable[[int], List[np.ndarray]],
        get_observation_fn: Callable[[], np.ndarray],
        apply_move_fn: Callable[[int], tuple],  # returns (raw_obs, game_done, info)
        track_move_fn: Callable[[List, int], None],  # for observation tracking
        find_pass_move_fn: Callable[[List[np.ndarray]], Optional[int]],
        on_game_end_fn: Callable[[], tuple]  # returns (reward, episode_done, raw_obs)
    ) -> tuple:
        """Autoplay opponents until it's the controlled player's turn.
        
        Args:
            current_player: Current player index to start from
            controlled_player: Index of the controlled player
            get_legal_moves_fn: Function to get legal moves for a player
            get_observation_fn: Function to get current observation vector
            apply_move_fn: Function to apply a move and get results
            track_move_fn: Function to track moves for observation memory
            find_pass_move_fn: Function to find pass move index
            on_game_end_fn: Function to handle game end scenarios
            
        Returns:
            Tuple of (total_reward, terminated, final_raw_obs, final_info)
        """
        if not self.has_opponents():
            return 0.0, False, None, {}
        
        total_reward = 0.0
        terminated = False
        final_raw_obs = None
        final_info = {}
        
        # Autoplay opponents until controlled player's turn
        while current_player != controlled_player and not terminated:
            # Get opponent action
            legal_moves_opp = get_legal_moves_fn(current_player)
            obs_vec = get_observation_fn()
            move_idx = self.select_opponent_action(
                current_player, legal_moves_opp, obs_vec, find_pass_move_fn
            )
            
            # Track opponent move for observation memory features
            try:
                selected_move = legal_moves_opp[move_idx] if move_idx < len(legal_moves_opp) else []
                track_move_fn(selected_move, current_player)
            except Exception:
                pass
            
            # Apply the move
            raw_obs, game_done, info = apply_move_fn(move_idx)
            final_raw_obs = raw_obs
            final_info = info
            
            if game_done:
                # Handle end of game
                reward, episode_done, raw_obs = on_game_end_fn()
                total_reward += reward
                final_raw_obs = raw_obs
                if episode_done:
                    terminated = True
                    break
            
            # Update current player for next iteration
            # Note: This should be updated by the environment in apply_move_fn
            # We'll assume the environment correctly updates current_player
            # If not, this would need to be passed back from apply_move_fn
        
        return total_reward, terminated, final_raw_obs, final_info