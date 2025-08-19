import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .bigtwo import ToyBigTwoFullRules


class BigTwoRLWrapper(gym.Env):
    """Stable-Baselines3 compatible wrapper for Big Two environment."""

    def __init__(
        self,
        num_players=4,
        games_per_episode=10,
        reward_function=None,
        controlled_player: int = 0,
        opponent_provider=None,
    ):
        super().__init__()
        self.env = ToyBigTwoFullRules(num_players)
        self.num_players = num_players
        self.games_per_episode = games_per_episode
        self.reward_function = (
            reward_function  # BaseReward instance for intermediate rewards
        )
        self.controlled_player = controlled_player
        # opponent_provider: callable or object providing opponents per episode.
        # It should expose `get_episode_opponents(num_players, controlled_player)` -> dict[int, BaseAgent]
        self.opponent_provider = opponent_provider
        self._episode_opponents = None  # lazily created per reset

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
        # New episode opponents (if provider is configured)
        if self.opponent_provider is not None:
            self._episode_opponents = self.opponent_provider.get_episode_opponents(
                self.num_players, self.controlled_player
            )
        else:
            self._episode_opponents = None
        self.current_obs = self._vectorize_obs(raw_obs)
        return self.current_obs, {}

    def step(self, action):
        info = {}
        reward_to_return = 0.0
        done = False
        terminated = False

        # Helper to apply a single move for the current player (either agent or opponent)
        def _apply_move(move_index: int):
            raw_obs_local, _rewards, game_done_local, info_local = self.env.step(
                move_index
            )
            self._cache_turn_counter += 1
            return raw_obs_local, game_done_local, info_local

        # If we have opponents configured, autoplay them until it's our turn
        if self._episode_opponents is not None:
            raw_obs = None
            # Autoplay opponents until controlled player's turn
            while (self.env.current_player != self.controlled_player) and (not done):
                # Opponent acts
                opp_player = self.env.current_player
                move_idx = self._select_opponent_action(opp_player)
                raw_obs, game_done, _ = _apply_move(move_idx)
                if game_done:
                    # Handle end of a single game
                    reward_to_return += self._handle_game_end_reward()
                    # Start next game or finish episode
                    ep_done, raw_obs = self._advance_or_reset_game()
                    if ep_done:
                        terminated = True
                        break
            if terminated:
                # Add episode bonus at episode end
                reward_to_return += float(self._calculate_episode_bonus())
                self.current_obs = self._vectorize_obs(raw_obs)
                return self.current_obs, reward_to_return, True, False, info

        # Now it should be the controlled player's turn (or no opponents configured)
        # Convert action index to actual move for current player
        legal_moves = self._get_legal_moves_cached(self.env.current_player)
        if action >= len(legal_moves):
            pass_idx = self._find_pass_move_idx(legal_moves)
            move_idx = pass_idx if pass_idx is not None else 0
        else:
            move_idx = action

        raw_obs, game_done, info = _apply_move(move_idx)

        if game_done:
            reward_to_return += self._handle_game_end_reward()
            ep_done, raw_obs = self._advance_or_reset_game()
            if ep_done:
                reward_to_return += float(self._calculate_episode_bonus())
                self.current_obs = self._vectorize_obs(raw_obs)
                return self.current_obs, reward_to_return, True, False, info

        # After our move, if opponents exist, autoplay them until it's our turn again or episode ends
        if self._episode_opponents is not None:
            while self.env.current_player != self.controlled_player:
                opp_player = self.env.current_player
                move_idx = self._select_opponent_action(opp_player)
                raw_obs, game_done, _ = _apply_move(move_idx)
                if game_done:
                    reward_to_return += self._handle_game_end_reward()
                    ep_done, raw_obs = self._advance_or_reset_game()
                    if ep_done:
                        reward_to_return += float(self._calculate_episode_bonus())
                        self.current_obs = self._vectorize_obs(raw_obs)
                        return self.current_obs, reward_to_return, True, False, info

        # Return observation for the controlled player's next decision point (or next game start)
        self.current_obs = self._vectorize_obs(raw_obs)
        return self.current_obs, reward_to_return, False, False, info

    def _vectorize_obs(self, raw_obs):
        """Convert dict observation to fixed 109-dim vector."""
        # Hand binary (52 features)
        hand_binary = raw_obs["hand"].astype(np.float32)

        # Last play binary (52 features)
        last_play_binary = raw_obs["last_play"].astype(np.float32)

        # Hand sizes for all players (4 features, padded with 0s if fewer players) - vectorized
        hand_sizes = np.zeros(4, dtype=np.float32)
        hand_sizes[: self.env.num_players] = np.sum(
            self.env.hands, axis=1, dtype=np.float32
        )

        # Last play exists flag (1 feature)
        last_play_exists = np.array([raw_obs["last_play_exists"]], dtype=np.float32)

        return np.concatenate(
            [hand_binary, last_play_binary, hand_sizes, last_play_exists]
        )

    def _get_simple_state_hash(self):
        """Create a lightweight hash of game state for cache invalidation."""
        # Use turn counter and hand sizes instead of expensive tuple creation
        hand_sizes = tuple(np.sum(self.env.hands, axis=1))
        last_play_len = np.sum(self.env.last_play[0]) if self.env.last_play else 0
        return (
            self.env.current_player,
            hand_sizes,
            last_play_len,
            self._cache_turn_counter,
        )

    def _get_legal_moves_cached(self, player):
        """Get legal moves with optimized caching to avoid duplicate computation."""
        current_state = self._get_simple_state_hash()

        # Check if cache is valid
        if (
            self._cache_player == player
            and self._cache_state_hash == current_state
            and self._cached_legal_moves is not None
        ):
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
                return self.reward_function.game_reward(
                    winner_player, player_idx, cards_left, all_cards_left
                )

        # Default reward if no custom function
        cards_left = int(np.sum(self.env.hands[player_idx]))
        if cards_left == 0:  # Winner
            return 1.0
        else:
            return -0.1 * cards_left  # Simple penalty

    def _calculate_episode_bonus(self):
        """Calculate episode bonus based on overall performance."""
        if self.reward_function is not None and self.games_played > 0:
            avg_cards_left = (
                (self.total_cards_when_losing / self.losses_count)
                if self.losses_count > 0
                else 0
            )
            return self.reward_function.episode_bonus(
                self.games_won, self.games_played, avg_cards_left
            )

        # Default episode bonus
        win_rate = self.games_won / self.games_played if self.games_played > 0 else 0
        return 0.5 if win_rate > 0.6 else 0

    # --- New helpers for opponent autoplay and episode/game transitions ---
    def _select_opponent_action(self, player_idx: int) -> int:
        """Select an action index for a non-controlled opponent player."""
        legal_moves = self._get_legal_moves_cached(player_idx)
        mask = np.zeros(2000, dtype=bool)
        for i in range(min(len(legal_moves), 2000)):
            mask[i] = True
        # Build observation for that player
        raw_obs = self.env._get_obs()
        obs_vec = self._vectorize_obs(raw_obs)
        agent = self._episode_opponents.get(player_idx)
        if agent is None:
            # Fallback: random legal
            legal_indices = np.where(mask)[0]
            return int(legal_indices[0]) if len(legal_indices) > 0 else 0
        action = agent.get_action(obs_vec, mask)
        if action >= len(legal_moves):
            # Clamp to a valid move (prefer PASS if available)
            pass_idx = self._find_pass_move_idx(legal_moves)
            return pass_idx if pass_idx is not None else 0
        return int(action)

    def _handle_game_end_reward(self) -> float:
        """Process end-of-game bookkeeping and return the controlled player's immediate reward."""
        self.games_played += 1
        # Reward for the controlled player
        reward = self._calculate_game_reward(self.controlled_player)
        if reward > 0:
            self.games_won += 1
        else:
            cards_left = np.sum(self.env.hands[self.controlled_player])
            self.total_cards_when_losing += cards_left
            self.losses_count += 1
        return float(reward)

    def _advance_or_reset_game(self):
        """Start the next game if episode not finished. Returns (episode_done, raw_obs)."""
        episode_done = self.games_played >= self.games_per_episode
        if episode_done:
            # Do NOT reset core env here to allow external evaluators to inspect final state
            # Episode bonus is handled by caller and the env will be reset on next external reset()
            return True, self.env._get_obs()
        # Clear caches for new game and reset core env
        self._cached_legal_moves = None
        self._cache_player = None
        self._cache_state_hash = None
        self._cache_turn_counter = 0
        raw_obs = self.env.reset()
        return False, raw_obs
