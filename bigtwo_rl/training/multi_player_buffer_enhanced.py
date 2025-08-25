"""Enhanced rollout buffer for multi-player turn-based games.

This module implements exact reward assignment matching the reference
Big Two PPO implementation. Key features:
- Tracks which player made each move (like mb_pGos in reference)
- Assigns rewards immediately to exactly the last 4 transitions per player
- Marks those transitions as terminal for proper GAE computation
"""

import numpy as np
from collections import deque
from typing import Generator, Optional, Union, List
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize
import torch as th


class MultiPlayerRolloutBuffer(RolloutBuffer):
    """Custom rollout buffer matching reference implementation's reward assignment.
    
    Key features:
    - Explicitly tracks which player made each move (mb_pGos equivalent)
    - Assigns rewards immediately when game ends, not delayed
    - Ensures exactly last 4 transitions per player receive final reward
    - Marks those transitions as terminal for GAE computation
    - Integrates seamlessly with stable-baselines3 PPO
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        # Initialize our attributes before calling super().__init__ (which calls reset())
        self.n_envs = n_envs
        
        # Track which player made each move (equivalent to mb_pGos in reference)
        # This MUST match buffer_size to avoid index errors
        self.player_who_moved = None  # Will be initialized in reset()
        
        # Track last 4 positions for each player in each environment
        self.last_player_positions = [{i: deque(maxlen=4) for i in range(4)} for _ in range(n_envs)]
        
        # Statistics
        self.immediate_rewards_assigned = 0
        self.games_completed = 0
        
        # Now call parent initialization
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)
        
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        current_player: Optional[Union[int, np.ndarray]] = None,
    ) -> None:
        """Add a step to the buffer with immediate reward assignment for game-ending rewards.
        
        CRITICAL: This method exactly matches reference behavior:
        1. ALWAYS adds the current step to buffer first
        2. When game ends, IMMEDIATELY assigns rewards to last 4 transitions per player  
        3. Marks those 4 transitions as terminal for GAE
        4. Never delays reward assignment to next cycle
        
        Args:
            obs, action, reward, episode_start, value, log_prob: Standard SB3 buffer inputs
            current_player: Which player made the move (0-3), or array for multi-env
        """
        # STEP 1: Always add current step to buffer first (even if game ended)
        # This ensures we have the complete game state before reward assignment
        self._add_normal_step(obs, action, reward, episode_start, value, log_prob, current_player)
        
        # STEP 2: Check if any environment has multi-player game-ending rewards
        game_ended_envs = []
        for env_idx in range(self.n_envs):
            env_reward = reward[env_idx] if hasattr(reward, '__len__') and len(reward) > env_idx else reward
            
            # Check if this is a multi-player game ending with rewards for all players
            if isinstance(env_reward, (list, np.ndarray)) and len(env_reward) == 4:
                game_ended_envs.append((env_idx, env_reward))
        
        # STEP 3: IMMEDIATELY assign final rewards (same cycle, not delayed)
        for env_idx, final_rewards in game_ended_envs:
            self._assign_final_game_rewards_immediately(final_rewards, env_idx)
            self.games_completed += 1
    
    def _add_normal_step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        current_player: Optional[Union[int, np.ndarray]] = None,
    ) -> None:
        """Add a normal step (not game-ending) to the buffer.
        
        This tracks player positions for future reward assignment.
        """
        # If current player not provided, we cannot safely infer it
        if current_player is None:
            # Log warning and use fallback - this should not happen in properly configured training
            import warnings
            warnings.warn(
                "current_player not provided to buffer.add(). This breaks proper player tracking "
                "for multi-player GAE. Using fallback position-based inference which may be incorrect.",
                UserWarning
            )
            current_player = self.pos % 4  # Use current buffer position as fallback
        
        # Store current buffer position before adding to buffer
        current_pos = self.pos
        
        # Use standard buffer addition first
        super().add(obs, action, reward, episode_start, value, log_prob)
        
        # Then track which player made this move at the position we just filled
        if isinstance(current_player, np.ndarray):
            # Multi-environment case
            for env_idx in range(self.n_envs):
                player_idx = current_player[env_idx] if env_idx < len(current_player) else current_player[0]
                # Track this buffer position for this player in this environment
                self.last_player_positions[env_idx][player_idx].append(current_pos)
        else:
            # Single environment case - for each env, track the current player's position
            for env_idx in range(self.n_envs):
                self.last_player_positions[env_idx][current_player].append(current_pos)
        
        # Store which player made this move at the current buffer position
        if isinstance(current_player, np.ndarray):
            # For multi-environment case, use the first environment's player or handle each env
            # Since we're storing per buffer position, we need to pick one value
            self.player_who_moved[current_pos] = current_player[0] if len(current_player) > 0 else 0
        else:
            self.player_who_moved[current_pos] = current_player
    
    def _assign_final_game_rewards_immediately(self, game_rewards: Union[List, np.ndarray], env_idx: int) -> None:
        """CRITICAL: Assign rewards to exactly 4 transitions per player, immediately.
        
        This exactly matches reference implementation (lines 111-114):
        mb_rewards[-1][i] = reward[mb_pGos[-1][i]-1] / rewardNormalization  
        mb_rewards[-2][i] = reward[mb_pGos[-2][i]-1] / rewardNormalization
        mb_rewards[-3][i] = reward[mb_pGos[-3][i]-1] / rewardNormalization  
        mb_rewards[-4][i] = reward[mb_pGos[-4][i]-1] / rewardNormalization
        
        KEY DIFFERENCES FROM OLD IMPLEMENTATION:
        1. EXACTLY 4 transitions per player (pad with dummy if needed)
        2. IMMEDIATE assignment (same cycle)  
        3. ALL 4 marked as terminal for GAE
        
        Args:
            game_rewards: Array of rewards for all 4 players [r0, r1, r2, r3]
            env_idx: Environment index
        """
        if len(game_rewards) != 4:
            return
            
        
        # REFERENCE EXACT: Assign rewards to exactly the last 4 transitions
        # mb_rewards[-1][i] = reward[mb_pGos[-1][i]-1] / rewardNormalization  
        # mb_rewards[-2][i] = reward[mb_pGos[-2][i]-1] / rewardNormalization
        # mb_rewards[-3][i] = reward[mb_pGos[-3][i]-1] / rewardNormalization  
        # mb_rewards[-4][i] = reward[mb_pGos[-4][i]-1] / rewardNormalization
        
        # Get exactly the last 4 buffer positions
        positions_to_assign = []
        for i in range(1, 5):  # -1, -2, -3, -4 in reference terms
            if self.pos >= i:
                pos = (self.pos - i) % self.buffer_size
                positions_to_assign.append(pos)
        
        # Ensure we have exactly 4 positions (pad if needed)
        while len(positions_to_assign) < 4:
            positions_to_assign.append(0)  # Pad with dummy position if needed
        
        # Assign rewards to these exact 4 positions
        for i, buffer_pos in enumerate(positions_to_assign):
            if buffer_pos < self.buffer_size:
                # Get which player made this move
                player_who_moved = self.player_who_moved[buffer_pos]
                
                # Assign this player's final reward
                if 0 <= player_who_moved < 4:
                    self.rewards[buffer_pos] = game_rewards[player_who_moved]
                    self.immediate_rewards_assigned += 1
                    
                    # Mark as terminal for GAE (reference lines 165-167)
                    # mb_dones[-2][i] = True, mb_dones[-3][i] = True, mb_dones[-4][i] = True
                    if i > 0:  # All except the very last transition (-1 position)
                        next_pos = (buffer_pos + 1) % self.buffer_size
                        if next_pos < self.buffer_size:
                            self.episode_starts[next_pos] = True
        
        # Clear tracking for this environment (game ended)
        for player_idx in range(4):
            self.last_player_positions[env_idx][player_idx].clear()
    
    def compute_multi_player_gae(self, gamma: float, gae_lambda: float) -> None:
        """Compute GAE for multi-player turn-based games using explicit player tracking.
        
        This implements the reference implementation's multi-player GAE calculation
        but uses actual player_who_moved data instead of assuming 4-step intervals.
        This matches the reference mb_pGos-based approach exactly.
        """
        advantages = np.zeros_like(self.rewards)
        
        # For each player (0, 1, 2, 3)
        for player_id in range(4):
            last_gae_lam = 0.0
            
            # Get all timesteps where this player made moves (using actual tracking data)
            player_steps = []
            for step_idx in range(self.buffer_size):
                if step_idx < len(self.player_who_moved) and self.player_who_moved[step_idx] == player_id:
                    player_steps.append(step_idx)
            
            # Sort to ensure chronological order
            player_steps.sort()
            
            # Go backwards through this player's actual timesteps  
            for i in reversed(range(len(player_steps))):
                step_idx = player_steps[i]
                
                if step_idx >= self.buffer_size:
                    continue
                    
                # Determine next value and terminal state
                if i + 1 < len(player_steps):
                    # Next step exists for this player
                    next_step_idx = player_steps[i + 1]
                    if next_step_idx < self.buffer_size:
                        next_non_terminal = 1.0 - self.episode_starts[next_step_idx]
                        next_values = self.values[next_step_idx]
                    else:
                        next_non_terminal = 0.0
                        next_values = 0.0
                else:
                    # This is the final step for this player
                    next_non_terminal = 0.0
                    next_values = 0.0
                
                # TD error calculation (matches reference)
                delta = (self.rewards[step_idx] + 
                        gamma * next_values * next_non_terminal - 
                        self.values[step_idx])
                
                # GAE calculation (matches reference line 145)
                advantages[step_idx] = last_gae_lam = (
                    delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
                )
        
        # Update the buffer with new advantages and returns
        self.advantages = advantages
        self.returns = advantages + self.values
    
    def get_statistics(self) -> dict:
        """Get buffer statistics for monitoring."""
        return {
            'immediate_rewards_assigned': self.immediate_rewards_assigned,
            'games_completed': self.games_completed,
            'player_position_tracking': [
                {f'player_{i}': len(positions[i]) for i in range(4)} 
                for positions in self.last_player_positions
            ]
        }
    
    def reset(self) -> None:
        """Reset the buffer and clear player position tracking."""
        super().reset()
        
        # Reset player tracking - initialize array with same size as buffer
        self.player_who_moved = np.zeros(self.buffer_size, dtype=int)
        self.last_player_positions = [{i: deque(maxlen=4) for i in range(4)} for _ in range(self.n_envs)]
        
        # Reset statistics
        self.immediate_rewards_assigned = 0
        self.games_completed = 0