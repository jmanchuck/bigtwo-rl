"""Multi-Player PPO implementation for turn-based games.

This module provides an enhanced PPO implementation that integrates:
- MultiPlayerRolloutBuffer for delayed reward assignment
- MultiPlayerGAECallback for proper turn-based GAE calculation
- Reference-compatible training loop
"""

import torch as th
import numpy as np
from typing import Optional, Union, Dict, Any, Type, Callable
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import explained_variance
from gymnasium import spaces

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
                # Buffer statistics available
        
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
                    self.logger.record("multiPlayer/immediate_rewards_assigned",
                                      buffer_stats['immediate_rewards_assigned'])
    
    def collect_rollouts(
        self,
        env,
        callback: BaseCallback,
        rollout_buffer,
        n_rollout_steps: int,
    ) -> bool:
        """Enhanced rollout collection with explicit player tracking.
        
        This method extracts current_player from environment info and passes it
        to the buffer for proper player tracking, matching reference mb_pGos behavior.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to PyTorch tensor or to TensorDict
                obs_tensor = th.as_tensor(self._last_obs, device=self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout termination properly if it exists in infos
            if infos is not None and len(infos) > 0:
                if "TimeLimit.truncated" in infos[0]:
                    terminal_obs = [info.get("terminal_observation") for info in infos]
                else:
                    terminal_obs = None
                
                # Extract current_player from info for proper tracking
                current_players = []
                for info in infos:
                    if isinstance(info, dict) and "current_player" in info:
                        current_players.append(info["current_player"])
                    else:
                        # Fallback to None - buffer will handle inference
                        current_players.append(None)
                
                # Convert to numpy array if all are valid
                if all(cp is not None for cp in current_players):
                    current_players = np.array(current_players, dtype=int)
                else:
                    # If some are missing, fill with environment-based inference
                    for i, cp in enumerate(current_players):
                        if cp is None:
                            current_players[i] = i % 4  # Environment index mod 4
                    current_players = np.array(current_players, dtype=int)
            else:
                terminal_obs = None
                # When no infos available, infer player from environment count
                # This handles initial rollout collection before any environment steps
                current_players = np.arange(self.n_envs) % 4  # Assume 4-player game

            # Add to buffer with explicit player tracking
            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                current_player=current_players,  # Pass player info to buffer
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(th.as_tensor(new_obs, device=self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True
    
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
        # Exclude callback from serialization (contains non-serializable objects)
        if exclude is None:
            exclude = set()
        else:
            exclude = set(exclude)
        exclude.add("multi_player_callback")
        
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
        
        # MultiPlayerPPO model and metadata saved
    
    @classmethod
    def load(cls, path, env=None, device="auto", custom_objects=None, print_system_info=False, 
             force_reset=True, **kwargs):
        """Load MultiPlayerPPO model."""
        # Load the base model
        model = super().load(path, env=env, device=device, custom_objects=custom_objects,
                           print_system_info=print_system_info, force_reset=force_reset, 
                           **kwargs)
        
        # Recreate the multi-player callback (was excluded from serialization)
        model.multi_player_callback = MultiPlayerGAECallback(verbose=0)
        model.multi_player_callback.model = model
        
        # Try to load metadata
        metadata_path = f"{path}_metadata.json"
        try:
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # MultiPlayerPPO metadata loaded
        except FileNotFoundError:
            # No metadata file found, using default settings
            pass
        
        return model