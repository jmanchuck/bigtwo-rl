"""Multi-Player PPO implementation for turn-based games.

This module provides an enhanced PPO implementation that integrates:
- MultiPlayerRolloutBuffer for delayed reward assignment
- MultiPlayerGAECallback for proper turn-based GAE calculation
- Reference-compatible training loop
"""

from collections import deque
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from stable_baselines3.common import utils
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule

from .callbacks import MultiPlayerGAECallback
from .multi_player_buffer_enhanced import MultiPlayerRolloutBuffer


class MultiPlayerPPO(MaskablePPO):
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
        policy: str | type[BasePolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 2048,
        batch_size: int | None = None,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: float | Schedule | None = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rollout_buffer_class: type[MultiPlayerRolloutBuffer] | None = None,
        rollout_buffer_kwargs: dict[str, Any] | None = None,
        target_kl: float | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
        league_opponent_prob: float = 0.0,
        snapshot_interval_rollouts: int = 20,
        snapshot_max_policies: int = 8,
    ):
        """Initialize MultiPlayerPPO.

        Args:
            All other args: Same as stable-baselines3 PPO

        """
        # Masked categorical distributions in long runs can hit tiny numerical
        # simplex violations; disabling strict validation avoids hard failures.
        th.distributions.Distribution.set_default_validate_args(False)

        if rollout_buffer_class is None:
            rollout_buffer_class = MultiPlayerRolloutBuffer

        # Ensure buffer kwargs include required parameters
        if rollout_buffer_kwargs is None:
            rollout_buffer_kwargs = {}

        # Set default batch_size if None (required for stable-baselines3 compatibility)
        if batch_size is None:
            batch_size = 64

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

        # League/self-play-vs-older-versions options (disabled by default).
        self.league_opponent_prob = float(league_opponent_prob)
        self.snapshot_interval_rollouts = int(snapshot_interval_rollouts)
        self.snapshot_max_policies = int(snapshot_max_policies)
        self._rollout_counter = 0
        self._policy_snapshots: deque = deque(maxlen=self.snapshot_max_policies)
        self._snapshot_policy = None
        self._rollout_carryover: dict[str, np.ndarray] | None = None

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
            if hasattr(self.rollout_buffer, "get_statistics"):
                buffer_stats = self.rollout_buffer.get_statistics()
                # Buffer statistics available

        # Ensure masks in rollout buffer are valid before PPO optimization.
        self._sanitize_rollout_buffer_masks()

        # Run standard PPO training
        super().train()

        # Periodically snapshot policy so opponents can be sampled from older versions.
        self._rollout_counter += 1
        if self.snapshot_interval_rollouts > 0 and self._rollout_counter % self.snapshot_interval_rollouts == 0:
            snap_state = {
                k: v.detach().clone().cpu()
                for k, v in self.policy.state_dict().items()
            }
            self._policy_snapshots.append(snap_state)

        # Log additional multi-player specific metrics
        if self.multi_player_callback:
            callback_stats = self.multi_player_callback.get_statistics()

            # Record to tensorboard if available
            if hasattr(self, "_logger") and self._logger is not None:
                self.logger.record("multiPlayer/gae_recalculations", callback_stats["gae_recalculations"])

                if hasattr(self.rollout_buffer, "get_statistics"):
                    buffer_stats = self.rollout_buffer.get_statistics()
                    self.logger.record("multiPlayer/games_completed", buffer_stats["games_completed"])
                    self.logger.record(
                        "multiPlayer/immediate_rewards_assigned", buffer_stats["immediate_rewards_assigned"],
                    )

    def collect_rollouts(
        self,
        env,
        callback: BaseCallback,
        rollout_buffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        """Enhanced rollout collection with explicit player tracking.

        This method extracts current_player from environment info and passes it
        to the buffer for proper player tracking, matching reference mb_pGos behavior.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        action_masks = None
        carryover = self._rollout_carryover
        rollout_buffer.reset()
        if carryover is not None and hasattr(rollout_buffer, "prime_with_carryover"):
            n_steps = int(rollout_buffer.prime_with_carryover(carryover))
        if use_masking and not is_masking_supported(env):
            raise ValueError("Environment does not support action masking")

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                # Convert to PyTorch tensor or to TensorDict
                obs_tensor = utils.obs_as_tensor(self._last_obs, self.device)
                if use_masking:
                    action_masks = self._sanitize_action_masks(get_action_masks(env))
                values = self.policy.predict_values(obs_tensor)

                # Optional league play: for opponent turns, occasionally sample action
                # from an older snapshot policy to increase opponent diversity.
                use_league = (
                    self.league_opponent_prob > 0.0
                    and len(self._policy_snapshots) > 0
                    and hasattr(env, "env_method")
                )

                if not use_league:
                    actions, _, log_probs = self.policy(obs_tensor, action_masks=action_masks)
                else:
                    try:
                        curr_players = np.array(env.env_method("get_current_player"), dtype=int)
                    except Exception:
                        curr_players = np.zeros(env.num_envs, dtype=int)

                    actions_list: list[th.Tensor] = []
                    log_probs_list: list[th.Tensor] = []
                    for i in range(env.num_envs):
                        obs_i = obs_tensor[i : i + 1]
                        mask_i = action_masks[i : i + 1] if action_masks is not None else None

                        # Treat non-player-0 seats as opponents and occasionally sample old policy.
                        if curr_players[i] != 0 and np.random.rand() < self.league_opponent_prob:
                            if self._snapshot_policy is None:
                                self._snapshot_policy = copy_policy = type(self.policy)(
                                    self.policy.observation_space,
                                    self.policy.action_space,
                                    self.lr_schedule,
                                    **self.policy_kwargs,
                                )
                                copy_policy = copy_policy.to(self.device)
                                copy_policy.set_training_mode(False)
                                self._snapshot_policy = copy_policy

                            snap_state = self._policy_snapshots[np.random.randint(len(self._policy_snapshots))]
                            self._snapshot_policy.load_state_dict(snap_state, strict=True)
                            opp_policy = self._snapshot_policy
                            dist = opp_policy.get_distribution(obs_i, action_masks=mask_i)
                            a_i = dist.get_actions(deterministic=False)
                            lp_i = dist.log_prob(a_i)
                        else:
                            dist = self.policy.get_distribution(obs_i, action_masks=mask_i)
                            a_i = dist.get_actions(deterministic=False)
                            lp_i = dist.log_prob(a_i)

                        actions_list.append(a_i)
                        log_probs_list.append(lp_i)

                    actions = th.cat(actions_list, dim=0)
                    log_probs = th.cat(log_probs_list, dim=0)

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

                # Extract actor identity from info for proper tracking
                current_players = []
                for info in infos:
                    if isinstance(info, dict) and "player_who_moved" in info and info["player_who_moved"] is not None:
                        current_players.append(info["player_who_moved"])
                    elif isinstance(info, dict) and "current_player" in info:
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
                action_masks=action_masks,
            )

            # If env emitted terminal per-player rewards, back-assign to last 4 turns.
            if infos is not None:
                for env_idx, info in enumerate(infos):
                    if (
                        isinstance(info, dict)
                        and "final_rewards" in info
                        and hasattr(rollout_buffer, "assign_final_game_rewards")
                    ):
                        rollout_buffer.assign_final_game_rewards(info["final_rewards"], env_idx=env_idx)

            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(th.as_tensor(new_obs, device=self.device))  # type: ignore[arg-type]

        if hasattr(rollout_buffer, "get_carryover"):
            max_carry = max(0, n_rollout_steps - 1)
            self._rollout_carryover = rollout_buffer.get_carryover(min(4, max_carry))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    @staticmethod
    def _sanitize_action_masks(action_masks: np.ndarray) -> np.ndarray:
        """Ensure masks are valid for masked categorical distributions.

        Some edge states can surface all-false mask rows. MaskablePPO cannot
        build a valid categorical distribution from an empty legal-action set,
        so we force a safe fallback (pass action id 0).
        """
        masks = np.asarray(action_masks).astype(bool)
        if masks.ndim == 1:
            masks = masks.reshape(1, -1)

        invalid_rows = ~masks.any(axis=1)
        if np.any(invalid_rows):
            masks[invalid_rows, 0] = True
        return masks

    def _sanitize_rollout_buffer_masks(self) -> None:
        """Normalize and fix mask rows stored in rollout buffer."""
        if not hasattr(self.rollout_buffer, "action_masks"):
            return
        masks = getattr(self.rollout_buffer, "action_masks", None)
        if masks is None:
            return

        arr = np.asarray(masks)
        if arr.size == 0:
            return

        # Flatten all but action dimension, sanitize, then restore shape.
        if arr.ndim < 2:
            arr = self._sanitize_action_masks(arr)
            self.rollout_buffer.action_masks = arr
            return

        last_dim = arr.shape[-1]
        flat = arr.reshape(-1, last_dim)
        flat = self._sanitize_action_masks(flat)
        self.rollout_buffer.action_masks = flat.reshape(arr.shape)

    def get_multi_player_statistics(self) -> dict[str, Any]:
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
        if hasattr(self.rollout_buffer, "get_statistics"):
            buffer_stats = self.rollout_buffer.get_statistics()
            stats.update({f"buffer_{k}": v for k, v in buffer_stats.items()})

        return stats

    def _setup_learn(
        self,
        total_timesteps: int,
        callback=None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "MultiPlayerPPO",
        progress_bar: bool = False,
    ):
        """Setup learning with multi-player enhancements."""
        self._rollout_carryover = None
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
            "buffer_class": self.rollout_buffer.__class__.__name__,
            "multi_player_statistics": self.get_multi_player_statistics(),
            # Save critical hyperparameters that might get lost
            "hyperparams": {
                "batch_size": self.batch_size,
                "n_steps": self.n_steps,
                "n_epochs": self.n_epochs,
                "learning_rate": self.learning_rate
                if isinstance(self.learning_rate, (int, float))
                else str(self.learning_rate),
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
            },
        }

        metadata_path = f"{path}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # MultiPlayerPPO model and metadata saved

    @classmethod
    def load(
        cls, path, env=None, device="auto", custom_objects=None, print_system_info=False, force_reset=True, **kwargs,
    ):
        """Load MultiPlayerPPO model."""
        # Try to load metadata first to get hyperparameters
        metadata_path = f"{path}_metadata.json"
        saved_hyperparams = {}
        try:
            import json

            with open(metadata_path) as f:
                metadata = json.load(f)
            saved_hyperparams = metadata.get("hyperparams", {})
            # MultiPlayerPPO metadata loaded
        except FileNotFoundError:
            # No metadata file found, will use defaults
            pass

        # Override kwargs with saved hyperparameters if available
        if "batch_size" not in kwargs and "batch_size" in saved_hyperparams:
            kwargs["batch_size"] = saved_hyperparams["batch_size"]

        # Load the base model
        model = super().load(
            path,
            env=env,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
            force_reset=force_reset,
            **kwargs,
        )

        # Recreate the multi-player callback (was excluded from serialization)
        model.multi_player_callback = MultiPlayerGAECallback(verbose=0)
        model.multi_player_callback.model = model

        return model
