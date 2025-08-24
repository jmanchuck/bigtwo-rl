"""Multi-Player Experience Buffer for Self-Play Training.

This module implements experience collection and management for true self-play
training where all 4 players contribute experiences to the training dataset.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque


class MultiPlayerExperienceBuffer:
    """Collects and manages experiences from all 4 players simultaneously.
    
    This buffer handles the complex task of collecting, storing, and preparing
    experiences from 4 different player perspectives for PPO training. It ensures
    proper batching and retroactive reward assignment for all players.
    
    Key Features:
    - 4x more training data (experiences from all 4 players)
    - Proper retroactive reward assignment
    - Efficient batch preparation for PPO
    - Player-specific experience tracking
    - Episode-level reward computation
    """
    
    def __init__(self, max_buffer_size: int = 10000):
        """Initialize multi-player experience buffer.
        
        Args:
            max_buffer_size: Maximum number of experiences to store per player
        """
        self.max_buffer_size = max_buffer_size
        
        # Per-player experience storage
        self.player_experiences = {i: deque(maxlen=max_buffer_size) for i in range(4)}
        
        # Temporary storage for current episode/game
        self.current_episode_experiences = {i: [] for i in range(4)}
        self.current_game_experiences = {i: [] for i in range(4)}
        
        # Episode tracking
        self.episode_count = 0
        self.game_count = 0
        
        # Metrics
        self.player_win_counts = defaultdict(int)
        self.player_game_counts = defaultdict(int)
        
    def add_step_experience(
        self,
        player_idx: int,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: Optional[np.ndarray] = None,
        done: bool = False,
        info: Optional[Dict[str, Any]] = None,
        action_log_prob: Optional[float] = None,
        value: Optional[float] = None
    ) -> None:
        """Add a single step experience for a specific player.
        
        Args:
            player_idx: Index of the player (0-3)
            observation: Current observation
            action: Action taken
            reward: Immediate reward
            next_observation: Next observation (optional)
            done: Whether episode/game is done
            info: Additional information dict
            action_log_prob: Log probability of action (for PPO)
            value: Value estimate (for PPO)
        """
        if player_idx not in range(4):
            raise ValueError(f"Invalid player_idx {player_idx}. Must be 0-3.")
        
        experience = {
            'observation': observation,
            'action': action,
            'reward': reward,
            'next_observation': next_observation,
            'done': done,
            'info': info or {},
            'action_log_prob': action_log_prob,
            'value': value,
            'player_idx': player_idx,
            'game_count': self.game_count,
            'episode_count': self.episode_count
        }
        
        # Add to current episode and game tracking
        self.current_episode_experiences[player_idx].append(experience)
        self.current_game_experiences[player_idx].append(experience)
    
    def add_multi_player_step(
        self,
        observations: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        next_observations: Optional[List[np.ndarray]] = None,
        dones: List[bool] = None,
        infos: Optional[List[Dict[str, Any]]] = None,
        action_log_probs: Optional[List[float]] = None,
        values: Optional[List[float]] = None
    ) -> None:
        """Add experiences for all 4 players from a single step.
        
        Args:
            observations: List of observations for all 4 players
            actions: List of actions for all 4 players
            rewards: List of rewards for all 4 players
            next_observations: List of next observations (optional)
            dones: List of done flags for all 4 players
            infos: List of info dicts for all 4 players
            action_log_probs: List of action log probabilities
            values: List of value estimates
        """
        if len(observations) != 4:
            raise ValueError(f"Expected 4 observations, got {len(observations)}")
        
        # Set defaults
        dones = dones or [False] * 4
        infos = infos or [{}] * 4
        next_observations = next_observations or [None] * 4
        action_log_probs = action_log_probs or [None] * 4
        values = values or [None] * 4
        
        # Add experience for each player
        for player_idx in range(4):
            self.add_step_experience(
                player_idx=player_idx,
                observation=observations[player_idx],
                action=actions[player_idx],
                reward=rewards[player_idx],
                next_observation=next_observations[player_idx],
                done=dones[player_idx],
                info=infos[player_idx],
                action_log_prob=action_log_probs[player_idx],
                value=values[player_idx]
            )
    
    def finalize_game(self, winner_player: Optional[int] = None) -> None:
        """Finalize current game and apply retroactive rewards.
        
        Args:
            winner_player: Index of winning player (None if no winner determined)
        """
        self.game_count += 1
        
        # Track winner statistics
        if winner_player is not None:
            self.player_win_counts[winner_player] += 1
        
        for player_idx in range(4):
            self.player_game_counts[player_idx] += 1
        
        # Apply retroactive rewards to game experiences
        self._apply_retroactive_game_rewards(winner_player)
        
        # Move game experiences to main buffer
        for player_idx in range(4):
            self.player_experiences[player_idx].extend(
                self.current_game_experiences[player_idx]
            )
            # Clear current game experiences
            self.current_game_experiences[player_idx] = []
    
    def finalize_episode(self) -> None:
        """Finalize current episode and compute episode bonuses."""
        self.episode_count += 1
        
        # Apply episode-level bonuses/adjustments
        self._apply_episode_bonuses()
        
        # Clear current episode tracking
        for player_idx in range(4):
            self.current_episode_experiences[player_idx] = []
    
    def _apply_retroactive_game_rewards(self, winner_player: Optional[int]) -> None:
        """Apply retroactive rewards based on game outcome.
        
        Args:
            winner_player: Index of winning player
        """
        # For each player, adjust rewards in current game based on outcome
        for player_idx in range(4):
            game_experiences = self.current_game_experiences[player_idx]
            
            if not game_experiences:
                continue
                
            # Calculate retroactive reward adjustment
            retroactive_bonus = self._calculate_retroactive_reward(
                player_idx, winner_player, game_experiences
            )
            
            # Apply to last few experiences (representing final game state impact)
            num_experiences_to_adjust = min(4, len(game_experiences))
            bonus_per_step = retroactive_bonus / num_experiences_to_adjust
            
            for i in range(-num_experiences_to_adjust, 0):
                game_experiences[i]['reward'] += bonus_per_step
                game_experiences[i]['retroactive_reward'] = bonus_per_step
    
    def _calculate_retroactive_reward(
        self,
        player_idx: int,
        winner_player: Optional[int],
        game_experiences: List[Dict[str, Any]]
    ) -> float:
        """Calculate retroactive reward for a player based on game outcome.
        
        Args:
            player_idx: Index of the player
            winner_player: Index of winning player
            game_experiences: List of experiences for this player
            
        Returns:
            Retroactive reward to apply
        """
        if winner_player is None or not game_experiences:
            return 0.0
        
        # Basic win/loss bonus
        if player_idx == winner_player:
            base_bonus = 1.0  # Winner bonus
        else:
            base_bonus = -0.25  # Loss penalty
        
        # Adjust based on game length (shorter games = better performance)
        game_length = len(game_experiences)
        length_factor = max(0.5, 1.0 - (game_length - 20) / 100)  # Normalize around 20 moves
        
        return base_bonus * length_factor
    
    def _apply_episode_bonuses(self) -> None:
        """Apply episode-level bonuses to experiences."""
        # Calculate episode-level statistics
        episode_wins = defaultdict(int)
        episode_games = defaultdict(int)
        
        for player_idx in range(4):
            for exp in self.current_episode_experiences[player_idx]:
                episode_games[player_idx] += 1
                if exp.get('info', {}).get('won_game', False):
                    episode_wins[player_idx] += 1
        
        # Apply episode bonus to each player's experiences
        for player_idx in range(4):
            if episode_games[player_idx] == 0:
                continue
                
            win_rate = episode_wins[player_idx] / episode_games[player_idx]
            episode_bonus = (win_rate - 0.25) * 0.1  # Bonus above expected 25% win rate
            
            # Apply bonus to recent experiences
            player_experiences = self.current_episode_experiences[player_idx]
            bonus_per_step = episode_bonus / max(1, len(player_experiences))
            
            for exp in player_experiences[-10:]:  # Last 10 experiences get episode bonus
                exp['reward'] += bonus_per_step
                exp['episode_bonus'] = bonus_per_step
    
    def get_training_batch(
        self, batch_size: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Get combined training batch from all players.
        
        Args:
            batch_size: Maximum batch size (None for all available)
            
        Returns:
            Dict containing combined training data
        """
        # Collect experiences from all players
        all_experiences = []
        for player_idx in range(4):
            all_experiences.extend(list(self.player_experiences[player_idx]))
        
        if not all_experiences:
            return self._empty_batch()
        
        # Limit batch size if specified
        if batch_size is not None and len(all_experiences) > batch_size:
            # Sample randomly to maintain diversity
            indices = np.random.choice(
                len(all_experiences), size=batch_size, replace=False
            )
            all_experiences = [all_experiences[i] for i in indices]
        
        # Convert to arrays
        return self._experiences_to_arrays(all_experiences)
    
    def get_player_specific_batch(
        self, player_idx: int, batch_size: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Get training batch for specific player.
        
        Args:
            player_idx: Index of player (0-3)
            batch_size: Maximum batch size (None for all available)
            
        Returns:
            Dict containing player-specific training data
        """
        if player_idx not in range(4):
            raise ValueError(f"Invalid player_idx {player_idx}")
        
        experiences = list(self.player_experiences[player_idx])
        
        if not experiences:
            return self._empty_batch()
        
        # Limit batch size if specified
        if batch_size is not None and len(experiences) > batch_size:
            experiences = experiences[-batch_size:]  # Take most recent
        
        return self._experiences_to_arrays(experiences)
    
    def _experiences_to_arrays(self, experiences: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Convert list of experiences to arrays for training.
        
        Args:
            experiences: List of experience dictionaries
            
        Returns:
            Dict with numpy arrays for training
        """
        if not experiences:
            return self._empty_batch()
        
        # Stack observations
        observations = np.stack([exp['observation'] for exp in experiences])
        actions = np.array([exp['action'] for exp in experiences])
        rewards = np.array([exp['reward'] for exp in experiences])
        dones = np.array([exp['done'] for exp in experiences])
        
        batch = {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'player_indices': np.array([exp['player_idx'] for exp in experiences])
        }
        
        # Add optional fields if available
        if experiences[0]['action_log_prob'] is not None:
            batch['action_log_probs'] = np.array([exp['action_log_prob'] for exp in experiences])
        
        if experiences[0]['value'] is not None:
            batch['values'] = np.array([exp['value'] for exp in experiences])
        
        if experiences[0]['next_observation'] is not None:
            batch['next_observations'] = np.stack([
                exp['next_observation'] for exp in experiences
                if exp['next_observation'] is not None
            ])
        
        return batch
    
    def _empty_batch(self) -> Dict[str, np.ndarray]:
        """Return empty batch with correct structure."""
        return {
            'observations': np.array([]),
            'actions': np.array([]),
            'rewards': np.array([]),
            'dones': np.array([]),
            'player_indices': np.array([])
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer and player statistics.
        
        Returns:
            Dict containing buffer statistics
        """
        total_experiences = sum(len(self.player_experiences[i]) for i in range(4))
        
        stats = {
            'total_experiences': total_experiences,
            'episodes_completed': self.episode_count,
            'games_completed': self.game_count,
            'experiences_per_player': {
                i: len(self.player_experiences[i]) for i in range(4)
            },
            'player_win_rates': {
                i: self.player_win_counts[i] / max(1, self.player_game_counts[i])
                for i in range(4)
            },
            'buffer_utilization': total_experiences / (4 * self.max_buffer_size)
        }
        
        return stats
    
    def clear(self) -> None:
        """Clear all stored experiences and reset counters."""
        self.player_experiences = {i: deque(maxlen=self.max_buffer_size) for i in range(4)}
        self.current_episode_experiences = {i: [] for i in range(4)}
        self.current_game_experiences = {i: [] for i in range(4)}
        self.episode_count = 0
        self.game_count = 0
        self.player_win_counts.clear()
        self.player_game_counts.clear()
    
    def __len__(self) -> int:
        """Return total number of experiences across all players."""
        return sum(len(self.player_experiences[i]) for i in range(4))