import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .bigtwo import ToyBigTwoFullRules


class BigTwoRLWrapper(gym.Env):
    """Stable-Baselines3 compatible wrapper for Big Two environment."""
    
    def __init__(self, num_players=4, games_per_episode=10, reward_function=None):
        super().__init__()
        self.env = ToyBigTwoFullRules(num_players)
        self.num_players = num_players
        self.games_per_episode = games_per_episode
        self.reward_function = reward_function  # BaseReward instance for intermediate rewards
        
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
        self.games_won = 0
        self.total_cards_when_losing = 0
        self.losses_count = 0
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Reset multi-game episode tracking
        self.games_played = 0
        self.games_won = 0
        self.total_cards_when_losing = 0
        self.losses_count = 0
        
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
            pass_idx = self._find_pass_move_idx(legal_moves)
            if pass_idx is not None:
                move_idx = pass_idx
            else:
                move_idx = 0
        else:
            move_idx = action
            
        raw_obs, rewards, game_done, info = self.env.step(move_idx)
        
        # Increment turn counter for cache invalidation
        self._cache_turn_counter += 1
        
        # Handle multi-game episodes with immediate rewards
        if game_done:
            self.games_played += 1
            
            # Calculate immediate game reward
            game_reward = self._calculate_game_reward(self.env.current_player - 1)  # Player rotated after step
            
            # Update episode statistics
            if game_reward > 0:  # Win
                self.games_won += 1
            else:  # Loss - track cards for episode bonus
                cards_left = np.sum(self.env.hands[self.env.current_player - 1])
                self.total_cards_when_losing += cards_left
                self.losses_count += 1
            
            # Check if episode is complete
            episode_done = self.games_played >= self.games_per_episode
            
            if episode_done:
                # Calculate episode bonus and add to final game reward
                episode_bonus = self._calculate_episode_bonus()
                final_reward = game_reward + episode_bonus
                self.current_obs = self._vectorize_obs(raw_obs)
                return self.current_obs, final_reward, True, False, info
            else:
                # Continue episode - return immediate game reward
                # Clear cache for new game
                self._cached_legal_moves = None
                self._cache_player = None
                self._cache_state_hash = None
                self._cache_turn_counter = 0
                
                raw_obs = self.env.reset()
                self.current_obs = self._vectorize_obs(raw_obs)
                return self.current_obs, game_reward, False, False, info
        
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
    
    def _find_pass_move_idx(self, legal_moves):
        """Find index of pass move (all-False array or empty list) in legal_moves list."""
        for i, move in enumerate(legal_moves):
            # Check for numpy array pass move (all False)
            if isinstance(move, np.ndarray):
                if not np.any(move):
                    return i
            # Check for list pass move (empty list)
            elif isinstance(move, list) and len(move) == 0:
                return i
        return None
    
    def _calculate_game_reward(self, player_idx):
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
                return self.reward_function.game_reward(winner_player, player_idx, cards_left, all_cards_left)
        
        # Default reward if no custom function
        cards_left = int(np.sum(self.env.hands[player_idx]))
        if cards_left == 0:  # Winner
            return 1.0
        else:
            return -0.1 * cards_left  # Simple penalty
    
    def _calculate_episode_bonus(self):
        """Calculate episode bonus based on overall performance."""
        if self.reward_function is not None and self.games_played > 0:
            avg_cards_left = (self.total_cards_when_losing / self.losses_count) if self.losses_count > 0 else 0
            return self.reward_function.episode_bonus(self.games_won, self.games_played, avg_cards_left)
        
        # Default episode bonus
        win_rate = self.games_won / self.games_played if self.games_played > 0 else 0
        return 0.5 if win_rate > 0.6 else 0