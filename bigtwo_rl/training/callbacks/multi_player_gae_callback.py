"""Multi-player GAE callback for turn-based games.

This callback recalculates GAE (Generalized Advantage Estimation) for 4-player
turn-based games like Big Two, where each player only acts every 4th timestep.
It implements the reference implementation's multi-player GAE calculation.
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional


class MultiPlayerGAECallback(BaseCallback):
    """Recalculates GAE for 4-player turn-based games.
    
    This callback implements the reference implementation's multi-player GAE
    calculation (lines 139-145 in mainBig2PPOSimulation.py):
    - Separate GAE calculation for each player position (0, 1, 2, 3)
    - Accounts for 4-step intervals between player actions
    - Properly handles terminal states and next values
    
    The callback integrates with MultiPlayerRolloutBuffer to ensure proper
    advantage estimation for multi-player environments.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.gae_recalculations = 0
        
    def _on_step(self) -> bool:
        """Called after each environment step. Return True to continue training."""
        return True
    
    def _on_rollout_end(self) -> bool:
        """Called after rollout collection, before training.
        
        This is the critical point where we recalculate advantages using
        the multi-player aware method instead of the default single-agent GAE.
        """
        # Check if we have the enhanced buffer with multi-player GAE capability
        rollout_buffer = self.model.rollout_buffer
        
        if hasattr(rollout_buffer, 'compute_multi_player_gae'):
            # Recalculating GAE for multi-player turn-based game
            # Use the buffer's multi-player GAE method
            rollout_buffer.compute_multi_player_gae(
                gamma=self.model.gamma,
                gae_lambda=self.model.gae_lambda
            )
            
            self.gae_recalculations += 1
            
            if self.verbose >= 2:
                # Log some statistics about the recalculation
                advantages = rollout_buffer.advantages
                returns = rollout_buffer.returns
                # GAE recalculation completed
                
        else:
            # Buffer doesn't support multi-player GAE
            pass
            
        return True
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        # Multi-player GAE recalculations completed
    
    def get_statistics(self) -> dict:
        """Get callback statistics for monitoring."""
        return {
            'gae_recalculations': self.gae_recalculations
        }