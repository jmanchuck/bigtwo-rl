import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .bigtwo import ToyBigTwoFullRules


class BigTwoRLWrapper(gym.Env):
    """Stable-Baselines3 compatible wrapper for Big Two environment."""
    
    def __init__(self, num_players=4, games_per_episode=10):
        super().__init__()
        self.env = ToyBigTwoFullRules(num_players)
        self.num_players = num_players
        self.games_per_episode = games_per_episode
        
        # Cache for legal moves to avoid duplicate computation
        self._cached_legal_moves = None
        self._cache_player = None
        self._cache_state_hash = None
        self._cache_turn_counter = 0
        
        # Fixed observation space: hand_binary(52) + last_play_binary(52) + hand_sizes(4) + last_play_exists(1)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(109,), dtype=np.float32
        )
        
        # Action space: Dynamic based on legal moves (max estimated around 1000 for 5-card combos)
        # We'll use a large fixed space and mask invalid actions
        self.action_space = spaces.Discrete(2000)
        
        # Track multi-game episode state
        self.current_obs = None
        self.games_played = 0
        self.cumulative_reward = 0.0
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Reset multi-game episode tracking
        self.games_played = 0
        self.cumulative_reward = 0.0
        
        # Clear cache on reset
        self._cached_legal_moves = None
        self._cache_player = None
        self._cache_state_hash = None
        self._cache_turn_counter = 0
        
        raw_obs = self.env.reset(seed=seed)
        self.current_obs = self._vectorize_obs(raw_obs)
        return self.current_obs, {}
    
    def step(self, action):
        # Convert action index to actual move
        legal_moves = self._get_legal_moves_cached(self.env.current_player)
        
        if action >= len(legal_moves):
            # Invalid action - force pass if legal, otherwise random legal move
            if [] in legal_moves:
                move_idx = legal_moves.index([])
            else:
                move_idx = 0
        else:
            move_idx = action
            
        raw_obs, rewards, game_done, info = self.env.step(move_idx)
        
        # Increment turn counter for cache invalidation
        self._cache_turn_counter += 1
        
        # Get reward for current player (before rotation)
        player_reward = rewards[self.env.current_player - 1]  # Player rotated after step
        
        # Handle multi-game episodes
        if game_done:
            self.games_played += 1
            self.cumulative_reward += player_reward
            
            # Check if episode is complete
            episode_done = self.games_played >= self.games_per_episode
            
            if episode_done:
                # Return average reward across all games
                final_reward = self.cumulative_reward / self.games_per_episode
                self.current_obs = self._vectorize_obs(raw_obs)
                return self.current_obs, final_reward, True, False, info
            else:
                # Start new game within episode
                # Clear cache for new game
                self._cached_legal_moves = None
                self._cache_player = None
                self._cache_state_hash = None
                self._cache_turn_counter = 0
                
                raw_obs = self.env.reset()
                self.current_obs = self._vectorize_obs(raw_obs)
                return self.current_obs, 0.0, False, False, info
        
        self.current_obs = self._vectorize_obs(raw_obs)
        return self.current_obs, 0.0, False, False, info  # No reward until episode ends
    
    def _vectorize_obs(self, raw_obs):
        """Convert dict observation to fixed 109-dim vector."""
        # Hand binary (52 features)
        hand_binary = raw_obs["hand"].astype(np.float32)
        
        # Last play binary (52 features)
        last_play_binary = raw_obs["last_play"].astype(np.float32)
        
        # Hand sizes for all players (4 features, padded with 0s if fewer players) - vectorized
        hand_sizes = np.zeros(4, dtype=np.float32)
        hand_sizes[:self.env.num_players] = np.sum(self.env.hands, axis=1, dtype=np.float32)
        
        # Last play exists flag (1 feature)
        last_play_exists = np.array([raw_obs["last_play_exists"]], dtype=np.float32)
        
        return np.concatenate([hand_binary, last_play_binary, hand_sizes, last_play_exists])
    
    def _get_simple_state_hash(self):
        """Create a lightweight hash of game state for cache invalidation."""
        # Use turn counter and hand sizes instead of expensive tuple creation
        hand_sizes = tuple(np.sum(self.env.hands, axis=1))
        last_play_len = np.sum(self.env.last_play[0]) if self.env.last_play else 0
        return (self.env.current_player, hand_sizes, last_play_len, self._cache_turn_counter)
    
    def _get_legal_moves_cached(self, player):
        """Get legal moves with optimized caching to avoid duplicate computation."""
        current_state = self._get_simple_state_hash()
        
        # Check if cache is valid
        if (self._cache_player == player and 
            self._cache_state_hash == current_state and 
            self._cached_legal_moves is not None):
            return self._cached_legal_moves
        
        # Cache miss - compute and store
        self._cached_legal_moves = self.env.legal_moves(player)
        self._cache_player = player
        self._cache_state_hash = current_state
        
        return self._cached_legal_moves
    
    def get_action_mask(self):
        """Return boolean mask for legal actions."""
        legal_moves = self._get_legal_moves_cached(self.env.current_player)
        mask = np.zeros(2000, dtype=bool)
        
        for i, move in enumerate(legal_moves):
            if i < 2000:  # Safety check
                mask[i] = True
                
        return mask