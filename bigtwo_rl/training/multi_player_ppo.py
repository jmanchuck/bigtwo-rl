"""Multi-Player PPO implementation for turn-based games.

This module provides an enhanced PPO implementation that integrates:
- MultiPlayerRolloutBuffer for delayed reward assignment
- MultiPlayerGAECallback for proper turn-based GAE calculation
- Reference-compatible training loop
"""

import torch as th
from typing import Optional, Union, Dict, Any, Type, Callable
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import explained_variance

from .multi_player_buffer_enhanced import MultiPlayerRolloutBuffer
from .callbacks import MultiPlayerGAECallback


class MultiPlayerPPO(PPO):
    """PPO enhanced for multi-player turn-based games.
    
    This class extends stable-baselines3 PPO with:
    - Automatic use of MultiPlayerRolloutBuffer for delayed reward assignment
    - Integration of MultiPlayerGAECallback for proper GAE calculation
    - Reference-compatible training loop that understands turn-based structure
    
    The API remains identical to standard PPO, but with multi-player enhancements
    active by default.
    """
    
    def __init__(
        self,
        policy: Union[str, Type[BasePolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = None,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Optional[Union[float, Schedule]] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[MultiPlayerRolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        """Initialize MultiPlayerPPO.
        
        Args:
            All other args: Same as stable-baselines3 PPO
        """
        if rollout_buffer_class is None:
            rollout_buffer_class = MultiPlayerRolloutBuffer
        
        # Ensure buffer kwargs include required parameters
        if rollout_buffer_kwargs is None:
            rollout_buffer_kwargs = {}
    
        # Initialize parent PPO with our enhanced buffer
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        
        # Create and store the multi-player GAE callback
        self.multi_player_callback = MultiPlayerGAECallback(verbose=verbose)
        # Set the callback's model reference
        self.multi_player_callback.model = self
    
    def train(self) -> None:
        """Enhanced training step with multi-player GAE recalculation.
        
        This method:
        1. Triggers multi-player GAE recalculation (if enabled)
        2. Runs standard PPO training
        3. Logs enhanced statistics
        """
        # Trigger multi-player GAE recalculation before training
        if self.multi_player_callback:
            self.multi_player_callback._on_rollout_end()
            
            # Log enhanced statistics
            if hasattr(self.rollout_buffer, 'get_statistics'):
                buffer_stats = self.rollout_buffer.get_statistics()
                if self.verbose >= 1:
                    print(f"📊 Buffer stats: {buffer_stats['games_completed']} games, "
                          f"{buffer_stats['delayed_rewards_assigned']} delayed rewards")
        
        # Run standard PPO training
        super().train()
        
        # Log additional multi-player specific metrics
        if self.multi_player_callback:
            callback_stats = self.multi_player_callback.get_statistics()
            
            # Record to tensorboard if available
            if hasattr(self, '_logger') and self._logger is not None:
                self.logger.record("multiPlayer/gae_recalculations", 
                                  callback_stats['gae_recalculations'])
                
                if hasattr(self.rollout_buffer, 'get_statistics'):
                    buffer_stats = self.rollout_buffer.get_statistics()
                    self.logger.record("multiPlayer/games_completed", 
                                      buffer_stats['games_completed'])
                    self.logger.record("multiPlayer/delayed_rewards_assigned",
                                      buffer_stats['delayed_rewards_assigned'])
    
    def collect_rollouts(
        self,
        env,
        callback: BaseCallback,
        rollout_buffer,
        n_rollout_steps: int,
    ) -> bool:
        """Enhanced rollout collection with multi-player awareness.
        
        This method maintains compatibility with stable-baselines3 while
        ensuring proper integration with our enhanced buffer.
        """
        # Use standard PPO rollout collection
        # The MultiPlayerRolloutBuffer handles the multi-player logic internally
        return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
    
    def get_multi_player_statistics(self) -> Dict[str, Any]:
        """Get multi-player specific training statistics.
        
        Returns:
            Dictionary with multi-player training metrics
        """
        stats = {}
        
        # Get callback statistics
        if self.multi_player_callback:
            callback_stats = self.multi_player_callback.get_statistics()
            stats.update({f"callback_{k}": v for k, v in callback_stats.items()})
        
        # Get buffer statistics
        if hasattr(self.rollout_buffer, 'get_statistics'):
            buffer_stats = self.rollout_buffer.get_statistics()
            stats.update({f"buffer_{k}": v for k, v in buffer_stats.items()})
        
        return stats
    
    def _setup_learn(
        self,
        total_timesteps: int,
        callback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "MultiPlayerPPO",
        progress_bar: bool = False,
    ):
        """Setup learning with multi-player enhancements."""
        # Use our custom tensorboard log name
        return super()._setup_learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=tb_log_name,
            progress_bar=progress_bar,
        )
    
    def save(self, path, exclude=None, include=None):
        """Save model with multi-player enhancement metadata."""
        # Save the model state
        super().save(path, exclude=exclude, include=include)
        
        # Save additional metadata about multi-player enhancements
        import json
        metadata = {
            'buffer_class': self.rollout_buffer.__class__.__name__,
            'multi_player_statistics': self.get_multi_player_statistics()
        }
        
        metadata_path = f"{path}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.verbose >= 1:
            print(f"💾 MultiPlayerPPO model saved to {path}")
            print(f"📊 Metadata saved to {metadata_path}")
    
    @classmethod
    def load(cls, path, env=None, device="auto", custom_objects=None, print_system_info=False, 
             force_reset=True, **kwargs):
        """Load MultiPlayerPPO model."""
        # Load the base model
        model = super().load(path, env=env, device=device, custom_objects=custom_objects,
                           print_system_info=print_system_info, force_reset=force_reset, 
                           **kwargs)
        
        # Try to load metadata
        metadata_path = f"{path}_metadata.json"
        try:
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            if model.verbose >= 1:
                print(f"📊 MultiPlayerPPO metadata loaded from {metadata_path}")
                print(f"   Buffer class: {metadata.get('buffer_class')}")
        except FileNotFoundError:
            if model.verbose >= 1:
                print("⚠️  No metadata file found, using default settings")
        
        return model