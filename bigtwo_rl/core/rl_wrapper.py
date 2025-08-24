"""Big Two RL Environment with True Self-Play Training.

This wrapper provides both single-player training (controlled player + opponents) and true self-play 
training (all 4 players use same network) with comprehensive logging and metrics.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple, List, Union
from .bigtwo import ToyBigTwoFullRules
from .episode_manager import EpisodeManager
from .observation_builder import ObservationConfig, ObservationVectorizer
import inspect


class BigTwoRLWrapper(gym.Env):
    """Gymnasium-compatible RL wrapper for Big Two with true self-play support.

    This wrapper provides a clean interface for training RL agents on Big Two using true self-play
    where all 4 players in the same game use the same neural network. This enables genuine 
    self-play dynamics, co-evolution, and 4x more training data per game.

    Key Features:
    - True self-play training (all 4 players use same network)
    - Multi-game episode training with configurable episode lengths  
    - Comprehensive logging: 27+ metrics for strategy analysis
    - Custom reward functions with move bonuses and episode bonuses
    - Game context tracking for strategic move evaluation
    - Move type tracking (singles, pairs, five-card hands)
    - Episode performance metrics and rankings

    Architecture:
    - EpisodeManager: Handles multi-game episode tracking and comprehensive metrics
    - ObservationVectorizer: Converts game state to configurable feature vectors
    - True multi-agent: Network decides for whichever player's turn it is
    """

    def __init__(
        self,
        observation_config: ObservationConfig,
        num_players: int = 4,
        games_per_episode: int = 10,
        reward_function = None,
        track_move_history: bool = False,
    ):
        """Initialize Big Two RL wrapper with true self-play.

        Args:
            observation_config: Observation configuration
            num_players: Number of players (must be 4 for Big Two)
            games_per_episode: Number of games per training episode
            reward_function: BaseReward instance for custom rewards
            track_move_history: Whether to track detailed move history
        """
        super().__init__()
        
        if num_players != 4:
            raise ValueError(f"Big Two requires exactly 4 players, got {num_players}")
            
        self.num_players = num_players
        self.games_per_episode = games_per_episode
        self.reward_function = reward_function
        self.track_move_history = track_move_history
        
        # Store config for lazy initialization (multiprocessing-safe)
        self.obs_config = observation_config
        
        # Initialize observation space immediately (needed for PPO model creation)
        temp_vectorizer = ObservationVectorizer(observation_config)
        self.observation_space = temp_vectorizer.gymnasium_space
        self.action_space = spaces.Discrete(2000)
        
        # Lazy initialization for game components (will be created in reset())
        self.game = None
        self.obs_vectorizer = None
        self.episode_manager = None
        
        # True self-play experience tracking
        self.player_experiences = []  # List of experiences for each player
        self.current_obs_per_player = {}  # Store observations when they act
        
        # Episode tracking
        self.games_completed = 0
        self.episode_complete = False
        
    def _ensure_initialized(self):
        """Ensure game components are initialized (multiprocessing-safe lazy init)."""
        if self.game is None:
            self.game = ToyBigTwoFullRules(self.num_players, self.track_move_history)
            
        if self.obs_vectorizer is None:
            self.obs_vectorizer = ObservationVectorizer(self.obs_config)
            
        if self.episode_manager is None:
            self.episode_manager = EpisodeManager(
                games_per_episode=self.games_per_episode,
                num_players=self.num_players,
                controlled_player=0,  # All players are "controlled" in self-play
            )
        
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to start a new multi-game episode.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            Tuple of (initial_observation, info_dict) from current player
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Ensure components are initialized (multiprocessing-safe)
        self._ensure_initialized()
            
        # Reset all components
        self.game.reset()
        self.obs_vectorizer.reset()
        self.episode_manager.reset_episode()
        
        # Reset true self-play tracking
        self.player_experiences = []
        self.current_obs_per_player = {}
        self.games_completed = 0
        self.episode_complete = False
        
        # Get initial observation for current player
        obs = self._get_observation(self.game.current_player)
        
        info = {
            'current_player': self.game.current_player,
            'games_completed': self.games_completed,
            'episode_complete': self.episode_complete,
            'legal_moves_count': len(self.game.legal_moves(self.game.current_player))
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step where the current player acts using the network.

        This implements true self-play: all 4 players use the same network.
        The network decides for whichever player's turn it is, enabling co-evolution
        and 4x more training data per game.

        Args:
            action: Action index for the current acting player

        Returns:
            Tuple of (observation, reward, terminated, truncated, info) for next player
        """
        # Ensure initialized
        self._ensure_initialized()
        
        if self.game.done:
            # Game already finished, check if episode is complete
            return self._handle_episode_completion()
            
        current_player = self.game.current_player
        
        # Store current observation before taking action (for experience collection)
        current_obs = self._get_observation(current_player)
        self.current_obs_per_player[current_player] = current_obs
        
        # Validate action is within legal moves bounds
        legal_moves = self.game.legal_moves(current_player)
        if action >= len(legal_moves):
            # Invalid action - force a valid action (e.g., pass or first legal move)
            if len(legal_moves) > 0:
                action = 0  # Take first legal move
            else:
                raise ValueError(f"No legal moves available for player {current_player}")
        
        # Track move metrics and bonuses BEFORE executing the move
        self._track_move_metrics_and_bonuses(legal_moves, action, current_player)
        
        # Execute action in the game
        game_obs_dict, rewards_list, game_done, info_dict = self.game.step(action)
        
        # Increment step counter for episode metrics
        self.episode_manager.increment_steps()
        
        # Store this player's experience for training
        reward = rewards_list[current_player] if len(rewards_list) > current_player else 0.0
        self.player_experiences.append({
            'player': current_player,
            'observation': current_obs,
            'action': action,
            'reward': reward,
            'done': game_done,
            'info': info_dict
        })
        
        if game_done:
            return self._handle_game_completion(rewards_list)
        else:
            # Game continues, return observation for next player
            next_player = self.game.current_player
            next_obs = self._get_observation(next_player)
            
            info = {
                'current_player': next_player,
                'previous_player': current_player,
                'games_completed': self.games_completed,
                'episode_complete': False,
                'legal_moves_count': len(self.game.legal_moves(next_player))
            }
            
            return next_obs, 0.0, False, False, info
    
    def _get_observation(self, player: int) -> np.ndarray:
        """Get observation for specific player using observation vectorizer.

        Args:
            player: Player index (0-3)

        Returns:
            Observation vector for the player
        """
        # Temporarily set current player to get observation from their perspective
        original_player = self.game.current_player
        self.game.current_player = player
        
        # Get raw observation from game engine
        raw_obs = self.game._get_obs()
        
        # Restore original current player
        self.game.current_player = original_player
        
        # Convert to observation vector
        return self.obs_vectorizer.vectorize(raw_obs, self.game)
    
    def _track_move_metrics_and_bonuses(
        self, legal_moves: List[np.ndarray], action: int, current_player: int
    ) -> None:
        """Track move bonuses and move types for comprehensive metrics.

        Args:
            legal_moves: List of legal moves available
            action: Index of the selected move
            current_player: Player making the move
        """
        if action >= len(legal_moves):
            return  # Invalid action, nothing to track
            
        selected_move = legal_moves[action]
        
        if isinstance(selected_move, np.ndarray):
            # Convert boolean mask to list of card indices
            move_cards = np.where(selected_move)[0].tolist()
            
            # Track move type for episode metrics
            self.episode_manager.track_move_type(move_cards)
            
            # Calculate and track move bonus if reward function supports it
            if (self.reward_function is not None and 
                hasattr(self.reward_function, "move_bonus")):
                
                # Build game context for strategic move evaluation
                game_context = self._build_game_context(current_player)
                
                # Call move_bonus with context (backward compatible)
                sig = inspect.signature(self.reward_function.move_bonus)
                if 'game_context' in sig.parameters:
                    move_bonus = self.reward_function.move_bonus(move_cards, game_context)
                else:
                    move_bonus = self.reward_function.move_bonus(move_cards)
                
                self.episode_manager.add_move_bonus(move_bonus)
    
    def _build_game_context(self, player_idx: int) -> Dict[str, Any]:
        """Build game context dictionary for move quality evaluation.

        Args:
            player_idx: Index of the player making the move
        
        Returns:
            Dict containing game state information for strategic evaluation
        """
        context = {}
        
        # Get player's remaining hand
        player_hand = self.game.hands[player_idx]
        remaining_hand = np.where(player_hand)[0].tolist()
        context['remaining_hand'] = remaining_hand
        
        # Get opponent card counts
        opponent_card_counts = []
        for i in range(self.num_players):
            if i != player_idx:
                card_count = int(np.sum(self.game.hands[i]))
                opponent_card_counts.append(card_count)
        context['opponent_card_counts'] = opponent_card_counts
        
        # Determine game phase based on card counts
        min_hand_size = min([int(np.sum(self.game.hands[i])) for i in range(self.num_players)])
        max_hand_size = max([int(np.sum(self.game.hands[i])) for i in range(self.num_players)])
        
        if max_hand_size > 10:
            game_phase = 'OPENING'
        elif min_hand_size <= 3:
            game_phase = 'ENDGAME'
        else:
            game_phase = 'MIDGAME'
        context['game_phase'] = game_phase
        
        # Get last play information
        last_play_strength = 1  # Default single card strength
        if self.game.last_play is not None:
            last_play_cards = np.where(self.game.last_play[0])[0]
            last_play_strength = len(last_play_cards)
        context['last_play_strength'] = last_play_strength
        
        # Get current turn and position info
        context['current_player'] = self.game.current_player
        context['controlled_player'] = player_idx  # In self-play, current player is "controlled"
        context['passes_in_row'] = self.game.passes_in_row
        
        # Get game progress metrics
        total_cards_dealt = self.num_players * 13  # Standard 52 cards, 4 players
        total_cards_remaining = sum([int(np.sum(self.game.hands[i])) for i in range(self.num_players)])
        cards_played_total = total_cards_dealt - total_cards_remaining
        context['cards_played_ratio'] = cards_played_total / total_cards_dealt if total_cards_dealt > 0 else 0
        
        return context
    
    def _handle_game_completion(self, rewards_list: List[float]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Handle game completion, process rewards, and check if episode should continue.

        Args:
            rewards_list: Final rewards for all players

        Returns:
            Tuple for next game or episode completion
        """
        self.games_completed += 1
        
        # Calculate cards left for all players at game end
        all_cards_left = [int(np.sum(self.game.hands[i])) for i in range(self.num_players)]
        
        # Process game end with episode manager (for controlled player = 0, but applies to all in self-play)
        winner_player, _ = self.episode_manager.handle_game_end(all_cards_left)
        
        # Apply custom reward function if provided
        if self.reward_function is not None:
            for i in range(self.num_players):
                processed_reward = self.reward_function.game_reward(
                    winner_player=winner_player,
                    player_idx=i, 
                    cards_left=all_cards_left[i],
                    all_cards_left=all_cards_left
                )
                rewards_list[i] = processed_reward
        
        # Store final rewards for all players in their experiences
        for exp in self.player_experiences:
            if exp['done']:  # This was the final action that ended the game
                exp['reward'] += rewards_list[exp['player']]
        
        # Check if episode is complete
        if self.games_completed >= self.games_per_episode:
            self.episode_complete = True
            return self._finalize_episode()
        else:
            # Start next game in episode
            self.game.reset()
            self.obs_vectorizer.reset()  # Reset observation memory for new game
            next_obs = self._get_observation(self.game.current_player)
            
            info = {
                'current_player': self.game.current_player,
                'games_completed': self.games_completed,
                'episode_complete': False,
                'legal_moves_count': len(self.game.legal_moves(self.game.current_player))
            }
            
            return next_obs, 0.0, False, False, info
    
    def _handle_episode_completion(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Handle when step is called but episode is already complete."""
        return self._finalize_episode()
        
    def _finalize_episode(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Finalize episode with episode bonus and comprehensive metrics."""
        # Calculate episode bonus using episode manager
        episode_bonus = float(self.episode_manager.calculate_episode_bonus(self.reward_function))
        
        # Return dummy observation and signal episode complete
        dummy_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Get comprehensive episode metrics for logging
        episode_metrics = self.episode_manager.get_episode_metrics()
        
        info = {
            'episode_complete': True,
            'games_completed': self.games_completed,
            'episode_bonus': episode_bonus,
            'multi_player_experiences': self.player_experiences,  # All experiences for training
            **episode_metrics  # Include all Big Two metrics for logging
        }
        
        return dummy_obs, episode_bonus, True, False, info
    
    def get_action_mask(self) -> np.ndarray:
        """Get action mask for current acting player.

        Returns:
            Boolean mask for legal actions (True = legal, False = illegal)
        """
        # Ensure initialized
        self._ensure_initialized()
        
        if self.game.done:
            # If game is done, no actions are legal
            return np.zeros(self.action_space.n, dtype=bool)
            
        legal_moves = self.game.legal_moves(self.game.current_player)
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        # Set legal action indices to True
        for i in range(min(len(legal_moves), self.action_space.n)):
            mask[i] = True
            
        return mask
    
    def action_masks(self) -> np.ndarray:
        """Alias for compatibility with ActionMasker wrapper."""
        return self.get_action_mask()
    
    def get_episode_metrics(self) -> Dict[str, float]:
        """Get episode metrics from episode manager.
        
        Returns:
            Dictionary of comprehensive Big Two episode metrics
        """
        if self.episode_manager is None:
            return {}
        return self.episode_manager.get_episode_metrics()
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render environment (basic game state info)."""
        if mode == "human":
            print(f"True Self-Play Big Two Game State:")
            print(f"Current Player: {self.game.current_player}")
            print(f"Games Completed: {self.games_completed}/{self.games_per_episode}")
            print(f"Game Done: {self.game.done}")
            if self.game.last_play is not None:
                cards_played = np.where(self.game.last_play[0])[0]
                print(f"Last Play: Player {self.game.last_play[1]} played cards {cards_played}")
            for i in range(4):
                cards_count = np.sum(self.game.hands[i])
                print(f"Player {i}: {cards_count} cards")
        return None
    
    def close(self) -> None:
        """Close environment (no resources to clean up)."""
        pass
    
    # Backward compatibility property for tournament/evaluation code
    @property
    def env(self):
        """Access to underlying game engine for backward compatibility."""
        self._ensure_initialized()
        return self.game
    
    # Backward compatibility properties to expose episode manager state
    @property
    def games_played(self) -> int:
        """Total games played in current episode."""
        return self.games_completed
    
    @property
    def games_won(self) -> int:
        """Number of games won in current episode (episode manager tracks this)."""
        if self.episode_manager is None:
            return 0
        return self.episode_manager.games_won
    
    @property
    def total_cards_when_losing(self) -> int:
        """Total cards remaining when losing games."""
        if self.episode_manager is None:
            return 0
        return self.episode_manager.total_cards_when_losing
    
    @property
    def losses_count(self) -> int:
        """Number of games lost in current episode."""
        if self.episode_manager is None:
            return 0
        return self.episode_manager.losses_count