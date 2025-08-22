import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple, List
from .bigtwo import ToyBigTwoFullRules
from .observation_builder import (
    ObservationConfig,
    ObservationVectorizer,
)


class BigTwoRLWrapper(gym.Env):
    """Stable-Baselines3 compatible wrapper for Big Two environment."""

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
        # opponent_provider: callable or object providing opponents per episode.
        # It should expose `get_episode_opponents(num_players, controlled_player)` -> dict[int, BaseAgent]
        self.opponent_provider = opponent_provider
        self._episode_opponents = None  # lazily created per reset

        # Set up observation configuration - must be explicit ObservationConfig instance
        self.obs_config = observation_config

        # Initialize observation vectorizer
        self.obs_vectorizer = ObservationVectorizer(self.obs_config)
        self.observation_space = self.obs_vectorizer.gymnasium_space

        # Cache for legal moves to avoid duplicate computation
        self._cached_legal_moves = None
        self._cache_player = None
        self._cache_state_hash = None
        self._cache_turn_counter = 0

        # Move bonus tracking for complex moves
        self._accumulated_move_bonuses = 0.0  # Bonuses from moves in current episode
        
        # Big Two metrics tracking
        self._episode_steps = 0
        self._move_counts = {"singles": 0, "pairs": 0, "five_cards": 0}
        self._final_positions = []  # Rankings across games in episode
        self._total_opponent_cards = 0  # For advantage calculation

        # Action space: Dynamic based on legal moves (max estimated around 1000 for 5-card combos)
        # We'll use a large fixed space and mask invalid actions
        self.action_space = spaces.Discrete(2000)

        # Track multi-game episode state
        self.current_obs = None
        self.games_played = 0
        self.games_won = 0
        self.total_cards_when_losing = 0
        self.losses_count = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)

        # Reset multi-game episode tracking
        self.games_played = 0
        self.games_won = 0
        self.total_cards_when_losing = 0
        self.losses_count = 0
        self._accumulated_move_bonuses = 0.0
        
        # Reset metrics tracking
        self._episode_steps = 0
        self._move_counts = {"singles": 0, "pairs": 0, "five_cards": 0}
        self._final_positions = []
        self._total_opponent_cards = 0

        # Clear cache on reset
        self._cached_legal_moves = None
        self._cache_player = None
        self._cache_state_hash = None
        self._cache_turn_counter = 0

        # Reset observation tracking
        self.obs_vectorizer.reset()

        raw_obs = self.env.reset(seed=seed)
        # New episode opponents (if provider is configured)
        if self.opponent_provider is not None:
            self._episode_opponents = self.opponent_provider.get_episode_opponents(
                self.num_players, self.controlled_player
            )
            # Provide env reference to opponents that support it (for better play)
            for agent in self._episode_opponents.values():
                if hasattr(agent, "set_env_reference"):
                    try:
                        agent.set_env_reference(self)  # type: ignore
                    except Exception:
                        pass
        else:
            self._episode_opponents = None
        self.current_obs = self._vectorize_obs(raw_obs)
        return self.current_obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
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
                legal_moves_opp = self._get_legal_moves_cached(opp_player)
                move_idx = self._select_opponent_action(opp_player)
                # Track opponent move for observation memory features
                try:
                    selected_move = legal_moves_opp[move_idx] if move_idx < len(legal_moves_opp) else []
                    self.obs_vectorizer.update_tracking(selected_move, opp_player, self.env)
                except Exception:
                    pass
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
                # Add Big Two metrics to info when episode completes
                info.update(self.get_episode_metrics())
                self.current_obs = self._vectorize_obs(raw_obs)
                return self.current_obs, reward_to_return, True, False, info

        # Now it should be the controlled player's turn (or no opponents configured)
        # Convert action index to actual move for current player
        legal_moves = self._get_legal_moves_cached(self.env.current_player)
        if len(legal_moves) == 0:
            move_idx = 0
        else:
            # Map oversized actions into current legal move range to avoid systematic passes
            move_idx = int(action) % len(legal_moves)

        # Track controlled player's move for observation memory features and metrics
        try:
            selected_move_cp = legal_moves[move_idx] if move_idx < len(legal_moves) else []
            self.obs_vectorizer.update_tracking(selected_move_cp, self.env.current_player, self.env)
        except Exception:
            pass
        raw_obs, game_done, info = _apply_move(move_idx)
        self._episode_steps += 1
        
        # Track move bonus and move types for controlled player
        if self.reward_function is not None and hasattr(self.reward_function, 'move_bonus'):
            selected_move = legal_moves[move_idx]
            if isinstance(selected_move, np.ndarray):
                # Convert boolean mask to list of card indices
                move_cards = np.where(selected_move)[0].tolist()
                move_bonus = self.reward_function.move_bonus(move_cards)
                self._accumulated_move_bonuses += move_bonus
                
                # Track move types for metrics
                move_count = len(move_cards)
                if move_count == 1:
                    self._move_counts["singles"] += 1
                elif move_count == 2:
                    self._move_counts["pairs"] += 1
                elif move_count == 5:
                    self._move_counts["five_cards"] += 1

        if game_done:
            reward_to_return += self._handle_game_end_reward()
            ep_done, raw_obs = self._advance_or_reset_game()
            if ep_done:
                reward_to_return += float(self._calculate_episode_bonus())
                # Add Big Two metrics to info when episode completes
                info.update(self.get_episode_metrics())
                self.current_obs = self._vectorize_obs(raw_obs)
                return self.current_obs, reward_to_return, True, False, info

        # After our move, if opponents exist, autoplay them until it's our turn again or episode ends
        if self._episode_opponents is not None:
            while self.env.current_player != self.controlled_player:
                opp_player = self.env.current_player
                legal_moves_opp = self._get_legal_moves_cached(opp_player)
                move_idx = self._select_opponent_action(opp_player)
                # Track opponent move for observation memory features
                try:
                    selected_move = legal_moves_opp[move_idx] if move_idx < len(legal_moves_opp) else []
                    self.obs_vectorizer.update_tracking(selected_move, opp_player, self.env)
                except Exception:
                    pass
                raw_obs, game_done, _ = _apply_move(move_idx)
                if game_done:
                    reward_to_return += self._handle_game_end_reward()
                    ep_done, raw_obs = self._advance_or_reset_game()
                    if ep_done:
                        reward_to_return += float(self._calculate_episode_bonus())
                        # Add Big Two metrics to info when episode completes
                        info.update(self.get_episode_metrics())
                        self.current_obs = self._vectorize_obs(raw_obs)
                        return self.current_obs, reward_to_return, True, False, info

        # Return observation for the controlled player's next decision point (or next game start)
        self.current_obs = self._vectorize_obs(raw_obs)
        return self.current_obs, reward_to_return, False, False, info

    def _vectorize_obs(self, raw_obs: Dict[str, Any]) -> np.ndarray:
        """Convert dict observation to configured feature vector."""
        return self.obs_vectorizer.vectorize(raw_obs, self.env)

    def _get_simple_state_hash(self) -> Tuple[int, Tuple[int, ...], int, int]:
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

    def _get_legal_moves_cached(self, player: int) -> List[np.ndarray]:
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

    def get_action_mask(self) -> np.ndarray:
        """Return boolean mask for legal actions."""
        legal_moves = self._get_legal_moves_cached(self.env.current_player)
        mask = np.zeros(2000, dtype=bool)

        for i, move in enumerate(legal_moves):
            if i < 2000:  # Safety check
                mask[i] = True

        return mask

    # For sb3-contrib MaskablePPO compatibility
    def action_masks(self) -> np.ndarray:
        """Alias used by ActionMasker wrapper to fetch the current legal action mask."""
        return self.get_action_mask()

    def _find_pass_move_idx(self, legal_moves: List[np.ndarray]) -> Optional[int]:
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

    def _calculate_episode_bonus(self) -> float:
        """Calculate episode bonus based on overall performance."""
        episode_bonus = 0.0
        
        if self.reward_function is not None and self.games_played > 0:
            avg_cards_left = (
                (self.total_cards_when_losing / self.losses_count)
                if self.losses_count > 0
                else 0
            )
            episode_bonus = self.reward_function.episode_bonus(
                self.games_won, self.games_played, avg_cards_left
            )
        else:
            # Default episode bonus
            win_rate = self.games_won / self.games_played if self.games_played > 0 else 0
            episode_bonus = 0.5 if win_rate > 0.6 else 0
        
        # Add accumulated move bonuses from complex moves throughout the episode
        total_bonus = episode_bonus + self._accumulated_move_bonuses
        return total_bonus
    
    def get_episode_metrics(self) -> Dict[str, float]:
        """Get Big Two-specific metrics for the completed episode."""
        if self.games_played == 0:
            return {}
        
        # Performance metrics
        win_rate = self.games_won / self.games_played
        avg_cards_remaining = self.total_cards_when_losing / self.losses_count if self.losses_count > 0 else 0
        # Overall average cards left including wins (winner has 0); closer to tournament stat
        # Compute by reconstructing per-game averages from tracked aggregates
        # We approximate overall average for the controlled player as:
        #   (sum(cards when losing) + 0 * games_won) / games_played
        avg_cards_overall = (
            (self.total_cards_when_losing) / self.games_played
            if self.games_played > 0
            else 0
        )
        final_position_avg = sum(self._final_positions) / len(self._final_positions) if self._final_positions else 0
        
        # Opponent analysis
        avg_opponent_cards_per_game = self._total_opponent_cards / (self.games_played * (self.num_players - 1))
        controlled_cards_per_game = self.total_cards_when_losing / self.losses_count if self.losses_count > 0 else 0
        cards_advantage = avg_opponent_cards_per_game - controlled_cards_per_game
        
        # Strategy metrics  
        total_moves = sum(self._move_counts.values())
        complex_moves = self._move_counts["pairs"] + self._move_counts["five_cards"]
        complex_move_ratio = complex_moves / total_moves if total_moves > 0 else 0
        avg_game_length = self._episode_steps / self.games_played
        
        # Count dominant wins (wins where average opponent had ≥8 cards)
        dominant_wins = sum(1 for pos in self._final_positions if pos == 1) if self.games_won > 0 else 0
        
        return {
            "bigtwo/win_rate": win_rate,
            "bigtwo/avg_cards_remaining": avg_cards_remaining,
            "bigtwo/avg_cards_overall": avg_cards_overall,
            "bigtwo/final_position_avg": final_position_avg,
            "bigtwo/cards_advantage": cards_advantage,
            "bigtwo/five_card_hands_played": float(self._move_counts["five_cards"]),
            "bigtwo/complex_move_ratio": complex_move_ratio,
            "bigtwo/move_bonuses_earned": self._accumulated_move_bonuses,
            "bigtwo/avg_game_length": avg_game_length,
            "bigtwo/games_completed": float(self.games_played),
            "bigtwo/games_won": float(self.games_won),
            "bigtwo/games_lost": float(self.losses_count),
        }

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
        
        # Calculate cards remaining for all players at game end
        all_cards_left = [int(np.sum(self.env.hands[i])) for i in range(self.num_players)]
        controlled_cards = all_cards_left[self.controlled_player]
        
        # Find the actual winner (player with 0 cards)
        winner_player = None
        for i, cards in enumerate(all_cards_left):
            if cards == 0:
                winner_player = i
                break
        
        # Track final position (1st place = 1, 4th place = 4) based on cards remaining
        sorted_positions = sorted(enumerate(all_cards_left), key=lambda x: x[1])
        position = next(i for i, (player_idx, _) in enumerate(sorted_positions) if player_idx == self.controlled_player) + 1
        self._final_positions.append(position)
        
        # Track opponent cards for advantage calculation
        opponent_cards = sum(cards for i, cards in enumerate(all_cards_left) if i != self.controlled_player)
        self._total_opponent_cards += opponent_cards
        
        # Reward calculation
        reward = self._calculate_game_reward(self.controlled_player)
        
        # Track wins/losses based on whether controlled player won
        if winner_player == self.controlled_player:
            self.games_won += 1
        else:
            # Only count cards when losing (not when winning)
            self.total_cards_when_losing += controlled_cards
            self.losses_count += 1
            
        return float(reward)

    def _advance_or_reset_game(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
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
        # Reset observation memory tracking per game
        try:
            self.obs_vectorizer.reset()
        except Exception:
            pass
        raw_obs = self.env.reset()
        return False, raw_obs
