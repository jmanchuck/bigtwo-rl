import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple, List
from .bigtwo import ToyBigTwoFullRules
from .episode_manager import EpisodeManager
from .opponent_controller import OpponentController
from .game_state_tracker import GameStateTracker
from .observation_builder import (
    ObservationConfig,
    ObservationVectorizer,
)


class BigTwoRLWrapper(gym.Env):
    """Gymnasium-compatible RL wrapper for Big Two card game environment.
    
    This wrapper provides a clean interface for training RL agents on Big Two,
    with modular components for episode management, opponent control, and game state tracking.
    
    Key Features:
    - Multi-game episode training with configurable episode lengths
    - Flexible opponent system with pluggable agent providers
    - Optimized legal move caching and action masking
    - Comprehensive episode metrics and statistics
    - Configurable observation features and reward functions
    
    Architecture:
    - EpisodeManager: Handles multi-game episode tracking and metrics
    - OpponentController: Manages opponent agents and autoplay logic  
    - GameStateTracker: Provides legal move caching and action validation
    - ObservationVectorizer: Converts game state to feature vectors
    """

    def __init__(
        self,
        observation_config: ObservationConfig,
        num_players=4,
        games_per_episode=10,
        reward_function=None,
        controlled_player: int = 0,
        opponent_provider=None,
        track_move_history: bool = False,
    ):
        super().__init__()
        self.env = ToyBigTwoFullRules(num_players, track_move_history=track_move_history)
        self.num_players = num_players
        self.games_per_episode = games_per_episode
        self.reward_function = (
            reward_function  # BaseReward instance for intermediate rewards
        )
        self.controlled_player = controlled_player
        
        # Initialize opponent controller
        self.opponent_controller = OpponentController(
            num_players=num_players,
            controlled_player=controlled_player,
            opponent_provider=opponent_provider
        )

        # Set up observation configuration - must be explicit ObservationConfig instance
        self.obs_config = observation_config

        # Initialize observation vectorizer
        self.obs_vectorizer = ObservationVectorizer(self.obs_config)
        self.observation_space = self.obs_vectorizer.gymnasium_space

        # Initialize game state tracker for caching and action masking
        self.game_state_tracker = GameStateTracker(action_space_size=2000)

        # Initialize episode manager for multi-game episode tracking
        self.episode_manager = EpisodeManager(
            games_per_episode=games_per_episode,
            num_players=num_players,
            controlled_player=controlled_player
        )

        # Action space: Dynamic based on legal moves (max estimated around 1000 for 5-card combos)
        # We'll use a large fixed space and mask invalid actions
        self.action_space = spaces.Discrete(2000)

        # Current observation state
        self.current_obs = None

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to start a new multi-game episode.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            Tuple of (initial_observation, info_dict)
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset episode manager
        self.episode_manager.reset_episode()

        # Reset game state tracker cache
        self.game_state_tracker.reset_cache()

        # Reset observation tracking
        self.obs_vectorizer.reset()

        raw_obs = self.env.reset(seed=seed)
        # Setup opponents for new episode
        self.opponent_controller.setup_episode_opponents(env_reference=self)
        self.current_obs = self._vectorize_obs(raw_obs)
        return self.current_obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step of the RL environment.
        
        Handles opponent autoplay, game transitions, and episode management.
        
        Args:
            action: Action index for the controlled player
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        info = {}
        reward_to_return = 0.0

        # Helper to apply a single move for the current player (either agent or opponent)
        def _apply_move(move_index: int):
            raw_obs_local, _rewards, game_done_local, info_local = self.env.step(
                move_index
            )
            self.game_state_tracker.increment_turn_counter()
            return raw_obs_local, game_done_local, info_local

        # Phase 1: Autoplay opponents until it's the controlled player's turn
        opp_reward, terminated, raw_obs, opp_info = self._autoplay_opponents_until_controlled_player(_apply_move)
        reward_to_return += opp_reward
        info.update(opp_info)
        
        if terminated:
            return self._handle_episode_termination(reward_to_return, raw_obs, info)

        # Phase 2: Process the controlled player's move
        raw_obs, game_done, step_info = self._process_controlled_player_move(action, _apply_move)
        info.update(step_info)

        # Phase 3: Handle game end if it occurred
        if game_done:
            reward_to_return += self._handle_game_end_reward()
            ep_done, raw_obs = self._advance_or_reset_game()
            if ep_done:
                return self._handle_episode_termination(reward_to_return, raw_obs, info)

        # Phase 4: Autoplay opponents after controlled player's move
        opp_reward_after, terminated_after, final_raw_obs, final_info = self._autoplay_opponents_after_controlled_move(_apply_move)
        reward_to_return += opp_reward_after
        info.update(final_info)
        
        if terminated_after:
            return self._handle_episode_termination(reward_to_return, final_raw_obs or raw_obs, info)

        # Return observation for the controlled player's next decision point
        self.current_obs = self._vectorize_obs(final_raw_obs or raw_obs)
        return self.current_obs, reward_to_return, False, False, info

    def _vectorize_obs(self, raw_obs: Dict[str, Any]) -> np.ndarray:
        """Convert dict observation to configured feature vector."""
        return self.obs_vectorizer.vectorize(raw_obs, self.env)

    def _execute_single_opponent_move(self, apply_move_fn) -> Tuple[Optional[Dict[str, Any]], bool, float]:
        """Execute a single opponent move and return results.
        
        Args:
            apply_move_fn: Function to apply a move to the environment
            
        Returns:
            Tuple of (raw_obs, game_done, additional_reward)
        """
        opp_player = self.env.current_player
        legal_moves_opp = self.game_state_tracker.get_legal_moves_cached(self.env, opp_player)
        move_idx = self._select_opponent_action(opp_player)
        
        # Track opponent move for observation memory features
        try:
            selected_move = legal_moves_opp[move_idx] if move_idx < len(legal_moves_opp) else []
            self.obs_vectorizer.update_tracking(selected_move, opp_player, self.env)
        except Exception:
            pass
            
        raw_obs, game_done, _ = apply_move_fn(move_idx)
        additional_reward = 0.0
        
        if game_done:
            # Handle end of a single game
            additional_reward = self._handle_game_end_reward()
            
        return raw_obs, game_done, additional_reward

    def _autoplay_opponents_until_controlled_player(self, apply_move_fn) -> Tuple[float, bool, Optional[Dict[str, Any]], Dict[str, Any]]:
        """Autoplay opponents until it's the controlled player's turn.
        
        Args:
            apply_move_fn: Function to apply a move to the environment
            
        Returns:
            Tuple of (total_reward, terminated, final_raw_obs, final_info)
        """
        if not self.opponent_controller.has_opponents():
            return 0.0, False, None, {}
            
        total_reward = 0.0
        terminated = False
        raw_obs = None
        info = {}
        
        # Autoplay opponents until controlled player's turn
        while (self.env.current_player != self.controlled_player) and (not terminated):
            raw_obs, game_done, additional_reward = self._execute_single_opponent_move(apply_move_fn)
            total_reward += additional_reward
            
            if game_done:
                # Start next game or finish episode
                ep_done, raw_obs = self._advance_or_reset_game()
                if ep_done:
                    terminated = True
                    break
                    
        return total_reward, terminated, raw_obs, info

    def _autoplay_opponents_after_controlled_move(self, apply_move_fn) -> Tuple[float, bool, Optional[Dict[str, Any]], Dict[str, Any]]:
        """Autoplay opponents after the controlled player's move until next controlled turn.
        
        Args:
            apply_move_fn: Function to apply a move to the environment
            
        Returns:
            Tuple of (total_reward, terminated, final_raw_obs, final_info)
        """
        if not self.opponent_controller.has_opponents():
            return 0.0, False, None, {}
            
        total_reward = 0.0
        terminated = False
        raw_obs = None
        info = {}
        
        while self.env.current_player != self.controlled_player:
            raw_obs, game_done, additional_reward = self._execute_single_opponent_move(apply_move_fn)
            total_reward += additional_reward
            
            if game_done:
                # Start next game or finish episode
                ep_done, raw_obs = self._advance_or_reset_game()
                if ep_done:
                    terminated = True
                    break
                    
        return total_reward, terminated, raw_obs, info

    def _handle_episode_termination(self, reward_to_return: float, raw_obs: Optional[Dict[str, Any]], info: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Handle episode termination by adding bonus and metrics.
        
        Args:
            reward_to_return: Accumulated reward so far
            raw_obs: Final raw observation
            info: Info dictionary to update
            
        Returns:
            Complete step return tuple for terminated episode
        """
        # Add episode bonus at episode end
        total_reward = reward_to_return + float(self.episode_manager.calculate_episode_bonus(self.reward_function))
        
        # Add Big Two metrics to info when episode completes
        info.update(self.episode_manager.get_episode_metrics())
        
        self.current_obs = self._vectorize_obs(raw_obs)
        return self.current_obs, total_reward, True, False, info

    def _calculate_episode_bonus(self) -> float:
        """Calculate episode bonus using episode manager."""
        return self.episode_manager.calculate_episode_bonus(self.reward_function)

    def get_episode_metrics(self) -> Dict[str, float]:
        """Get episode metrics from episode manager."""
        return self.episode_manager.get_episode_metrics()

    def _process_controlled_player_move(self, action: int, apply_move_fn) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        """Process the controlled player's move including validation and tracking.
        
        Args:
            action: Action index for the controlled player
            apply_move_fn: Function to apply a move to the environment
            
        Returns:
            Tuple of (raw_obs, game_done, info)
        """
        # Convert action index to actual move for current player
        legal_moves = self.game_state_tracker.get_legal_moves_cached(self.env, self.env.current_player)
        # Validate and clamp action to legal range
        move_idx = self.game_state_tracker.validate_and_clamp_action(action, legal_moves)

        # Track controlled player's move for observation memory features and metrics
        try:
            selected_move_cp = legal_moves[move_idx] if move_idx < len(legal_moves) else []
            self.obs_vectorizer.update_tracking(selected_move_cp, self.env.current_player, self.env)
        except Exception:
            pass
            
        raw_obs, game_done, info = apply_move_fn(move_idx)
        self.episode_manager.increment_steps()
        
        # Track move bonus and move types for controlled player
        self._track_move_bonus_and_metrics(legal_moves, move_idx)
        
        return raw_obs, game_done, info

    def _track_move_bonus_and_metrics(self, legal_moves: List[np.ndarray], move_idx: int) -> None:
        """Track move bonus and move types for the controlled player.
        
        Args:
            legal_moves: List of legal moves available
            move_idx: Index of the selected move
        """
        if self.reward_function is not None and hasattr(self.reward_function, 'move_bonus'):
            selected_move = legal_moves[move_idx]
            if isinstance(selected_move, np.ndarray):
                # Convert boolean mask to list of card indices
                move_cards = np.where(selected_move)[0].tolist()
                move_bonus = self.reward_function.move_bonus(move_cards)
                self.episode_manager.add_move_bonus(move_bonus)
                
                # Track move types for metrics
                self.episode_manager.track_move_type(move_cards)


    def get_action_mask(self) -> np.ndarray:
        """Return boolean mask for legal actions."""
        return self.game_state_tracker.get_action_mask(self.env, self.env.current_player)

    # For sb3-contrib MaskablePPO compatibility
    def action_masks(self) -> np.ndarray:
        """Alias used by ActionMasker wrapper to fetch the current legal action mask."""
        return self.get_action_mask()


    def _calculate_game_reward(self, player_idx: int) -> float:
        """Calculate immediate reward after game completion."""
        if self.reward_function is not None:
            # Find winner and get all cards left
            winner_player = None
            all_cards_left = []
            for p in range(self.num_players):
                cards_left = int(np.sum(self.env.hands[p]))
                all_cards_left.append(cards_left)
                if cards_left == 0:
                    winner_player = p

            if winner_player is not None:
                cards_left = all_cards_left[player_idx]
                return self.reward_function.game_reward(
                    winner_player, player_idx, cards_left, all_cards_left
                )

        # Default reward if no custom function
        cards_left = int(np.sum(self.env.hands[player_idx]))
        if cards_left == 0:  # Winner
            return 1.0
        else:
            return -0.1 * cards_left  # Simple penalty


    # --- New helpers for opponent autoplay and episode/game transitions ---
    def _select_opponent_action(self, player_idx: int) -> int:
        """Select an action index for a non-controlled opponent player."""
        legal_moves = self.game_state_tracker.get_legal_moves_cached(self.env, player_idx)
        # Build observation for that player
        raw_obs = self.env._get_obs()
        obs_vec = self._vectorize_obs(raw_obs)
        
        return self.opponent_controller.select_opponent_action(
            player_idx=player_idx,
            legal_moves=legal_moves,
            observation_vector=obs_vec,
            find_pass_move_fn=self.game_state_tracker.find_pass_move_idx
        )

    def _handle_game_end_reward(self) -> float:
        """Process end-of-game bookkeeping and return the controlled player's immediate reward."""
        # Calculate cards remaining for all players at game end
        all_cards_left = [int(np.sum(self.env.hands[i])) for i in range(self.num_players)]
        
        # Use episode manager to handle game end tracking
        winner_player, controlled_player_won = self.episode_manager.handle_game_end(all_cards_left)
        
        # Calculate reward using existing method
        reward = self._calculate_game_reward(self.controlled_player)
        
        return float(reward)

    def _advance_or_reset_game(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Start the next game if episode not finished. Returns (episode_done, raw_obs)."""
        episode_done = self.episode_manager.is_episode_complete()
        if episode_done:
            # Do NOT reset core env here to allow external evaluators to inspect final state
            # Episode bonus is handled by caller and the env will be reset on next external reset()
            return True, self.env._get_obs()
        # Clear caches for new game and reset core env
        self.game_state_tracker.clear_cache_for_new_game()
        # Reset observation memory tracking per game
        try:
            self.obs_vectorizer.reset()
        except Exception:
            pass
        raw_obs = self.env.reset()
        return False, raw_obs
    
    # Backward compatibility properties to expose episode manager state
    @property
    def games_played(self) -> int:
        """Number of games played in current episode."""
        return self.episode_manager.games_played
    
    @property
    def games_won(self) -> int:
        """Number of games won in current episode."""
        return self.episode_manager.games_won
    
    @property
    def total_cards_when_losing(self) -> int:
        """Total cards remaining when losing games."""
        return self.episode_manager.total_cards_when_losing
    
    @property
    def losses_count(self) -> int:
        """Number of games lost in current episode."""
        return self.episode_manager.losses_count
