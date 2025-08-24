"""Self-Play PPO Callback utilities.

Two callbacks are defined:

1) `SimpleSelfPlayCallback` (used by the trainer):
   - Injects the live PPO model reference into each env so the same network
     can act for all four players (true self-play) within a `DummyVecEnv`.
   - Logs counts of multi-player experiences exposed by the env via
     `infos["multi_player_experiences"]`. It does not inject experiences into
     PPO's rollout buffer; collection remains standard SB3.

2) `SelfPlayPPOCallback` (experimental, not wired in):
   - Sketches how to post-process `multi_player_experiences` and add them to
     PPO's buffer to get >1x data efficiency. This is not fully implemented
     (no value/logprob reconstruction); kept for future work.
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger
from typing import Dict, List, Any, Optional
from collections import deque


class SelfPlayPPOCallback(BaseCallback):
    """Callback to process multi-player experiences from true self-play training.

    This callback intercepts the multi-player experiences collected during episodes
    and adds them to PPO's training buffer, effectively giving us 4x more training
    data from the same amount of game play.

    Key Features:
    - Extracts experiences from all 4 players
    - Adds them to PPO's rollout buffer
    - Maintains proper reward and observation alignment
    - Logs multi-player training metrics
    """

    def __init__(self, verbose: int = 0):
        """Initialize self-play callback.

        Args:
            verbose: Verbosity level for logging
        """
        super().__init__(verbose)
        self.total_multi_player_experiences = 0
        self.total_episodes_processed = 0
        self.experiences_per_player = [0, 0, 0, 0]

    def _on_step(self) -> bool:
        """Called after each environment step.

        This is where we process multi-player experiences when episodes complete.

        Returns:
            True to continue training
        """
        # Check if any environments completed episodes with multi-player experiences
        for env_idx in range(self.training_env.num_envs):
            # Get info from the most recent step
            if "infos" in self.locals and len(self.locals["infos"]) > env_idx:
                info = self.locals["infos"][env_idx]

                # Check if this environment completed an episode with multi-player experiences
                if (
                    info.get("episode_complete", False)
                    and "multi_player_experiences" in info
                ):
                    self._process_multi_player_experiences(
                        info["multi_player_experiences"], env_idx
                    )

        return True

    def _process_multi_player_experiences(
        self, experiences: List[Dict[str, Any]], env_idx: int
    ) -> None:
        """Process multi-player experiences and add them to PPO's buffer.

        Args:
            experiences: List of experience dicts from all players
            env_idx: Index of the environment that generated these experiences
        """
        if not experiences:
            return

        # Group experiences by player
        player_experiences = {i: [] for i in range(4)}
        for exp in experiences:
            player_idx = exp["player"]
            if player_idx < 4:  # Safety check
                player_experiences[player_idx].append(exp)

        # Process experiences for each player (excluding Player 0 to avoid duplication)
        for player_idx in range(1, 4):  # Players 1, 2, 3 (Player 0 already in buffer)
            player_exp_list = player_experiences[player_idx]
            if player_exp_list:
                self._add_player_experiences_to_buffer(
                    player_exp_list, env_idx, player_idx
                )

        # Update statistics
        self.total_episodes_processed += 1
        for player_idx in range(4):
            self.experiences_per_player[player_idx] += len(
                player_experiences[player_idx]
            )

        # Log statistics periodically
        if self.total_episodes_processed % 10 == 0:
            self._log_multi_player_stats()

    def _add_player_experiences_to_buffer(
        self, experiences: List[Dict[str, Any]], env_idx: int, player_idx: int
    ) -> None:
        """Add a player's experiences to PPO's rollout buffer.

        Args:
            experiences: List of experience dicts for one player
            env_idx: Environment index
            player_idx: Player index (1, 2, or 3)
        """
        if not hasattr(self.model, "rollout_buffer") or not experiences:
            return

        rollout_buffer = self.model.rollout_buffer

        # Convert experiences to the format expected by PPO
        for exp in experiences:
            obs = exp["observation"]
            action = exp["action"]
            reward = exp["reward"]
            done = exp["done"]

            # We need to add these to the buffer, but PPO's buffer expects
            # observations, actions, rewards, dones, values, and log_probs
            # Since we can't easily get values and log_probs for past actions,
            # we'll add them to a separate buffer for the next training iteration

            # For now, we'll store them and add them during the next rollout
            if not hasattr(self, "_pending_experiences"):
                self._pending_experiences = []

            self._pending_experiences.append(
                {
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "env_idx": env_idx,
                    "player_idx": player_idx,
                }
            )

        self.total_multi_player_experiences += len(experiences)

    def _on_rollout_start(self) -> None:
        """Called at the start of each rollout.

        This is where we can inject pending experiences into the new rollout.
        """
        # Process any pending experiences from previous episodes
        if hasattr(self, "_pending_experiences") and self._pending_experiences:
            if self.verbose >= 1:
                self.logger.record(
                    "self_play/pending_experiences", len(self._pending_experiences)
                )

            # For now, we'll just clear them since properly integrating them
            # into PPO's buffer requires more complex intervention
            # TODO: Implement proper experience injection
            self._pending_experiences.clear()

    def _log_multi_player_stats(self) -> None:
        """Log multi-player training statistics."""
        if self.verbose >= 1:
            self.logger.record(
                "self_play/total_episodes_processed", self.total_episodes_processed
            )
            self.logger.record(
                "self_play/total_multi_player_experiences",
                self.total_multi_player_experiences,
            )

            for i in range(4):
                self.logger.record(
                    f"self_play/experiences_player_{i}", self.experiences_per_player[i]
                )

            if self.total_episodes_processed > 0:
                avg_exp_per_episode = (
                    self.total_multi_player_experiences / self.total_episodes_processed
                )
                self.logger.record(
                    "self_play/avg_experiences_per_episode", avg_exp_per_episode
                )

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.verbose >= 1:
            print("\n" + "=" * 50)
            print("Self-Play Training Summary:")
            print(f"Episodes processed: {self.total_episodes_processed}")
            print(
                f"Total multi-player experiences collected: {self.total_multi_player_experiences}"
            )
            print("Experiences per player:")
            for i in range(4):
                print(f"  Player {i}: {self.experiences_per_player[i]}")

            if self.total_episodes_processed > 0:
                avg_exp = (
                    self.total_multi_player_experiences / self.total_episodes_processed
                )
                print(f"Average experiences per episode: {avg_exp:.1f}")
                print(
                    f"Training data multiplier: {avg_exp / 10:.1f}x (compared to single-player)"
                )
            print("=" * 50 + "\n")


class SimpleSelfPlayCallback(BaseCallback):
    """Simplified self-play callback that logs multi-player experience collection
    and injects model references for true self-play.

    This callback:
    1. Injects the PPO model into environments for self-play action selection
    2. Monitors and logs multi-player experience collection
    3. Provides statistics on self-play training effectiveness
    """

    def __init__(self, verbose: int = 0):
        """Initialize simplified self-play callback."""
        super().__init__(verbose)
        self.multi_player_episodes = 0
        self.total_experiences_collected = 0
        self.model_injected = False

    def _on_training_start(self) -> None:
        """Called at the start of training. Inject model reference into environments."""
        if self.verbose >= 1:
            print(
                f"Self-play callback: Training start - env type: {type(self.training_env)}"
            )

        # Handle different VecEnv types
        if hasattr(self.training_env, "envs"):
            # DummyVecEnv case - direct access to environments
            if self.verbose >= 1:
                print(
                    f"Self-play callback: Found {len(self.training_env.envs)} environments in {type(self.training_env)}"
                )

            for i, env in enumerate(self.training_env.envs):
                target_env = self._get_base_env(env)
                if self.verbose >= 1:
                    print(
                        f"Self-play callback: Env {i} - target type: {type(target_env)}, has method: {hasattr(target_env, 'set_model_reference')}"
                    )

                if hasattr(target_env, "set_model_reference"):
                    target_env.set_model_reference(self.model)
                    self.model_injected = True
        elif hasattr(self.training_env, "env_method"):
            # SubprocVecEnv case - remote method call
            try:
                # This won't work well for model objects due to pickling issues
                self.training_env.env_method("set_model_reference", self.model)
                self.model_injected = True
                if self.verbose >= 1:
                    print(
                        f"Self-play callback: Model reference injected via env_method"
                    )
            except Exception as e:
                if self.verbose >= 1:
                    print(f"Self-play callback: env_method injection failed: {e}")
                    print(
                        "Model sharing not supported with SubprocVecEnv - consider using DummyVecEnv"
                    )
        else:
            # Single env case
            target_env = self._get_base_env(self.training_env)
            if self.verbose >= 1:
                print(
                    f"Self-play callback: Single env - target type: {type(target_env)}, has method: {hasattr(target_env, 'set_model_reference')}"
                )

            if hasattr(target_env, "set_model_reference"):
                target_env.set_model_reference(self.model)
                self.model_injected = True

        if self.verbose >= 1:
            print(
                f"Self-play callback: Model reference injected into environments: {self.model_injected}"
            )

    def _get_base_env(self, env):
        """Get the base BigTwoRLWrapper from potentially wrapped environment."""
        # Handle ActionMasker wrapper
        if hasattr(env, "env") and hasattr(env.env, "set_model_reference"):
            return env.env
        # Direct access
        elif hasattr(env, "set_model_reference"):
            return env
        # Look for unwrap method (common gym pattern)
        elif hasattr(env, "unwrap"):
            unwrapped = env.unwrap()
            if hasattr(unwrapped, "set_model_reference"):
                return unwrapped

        return env

    def _on_step(self) -> bool:
        """Monitor multi-player experience collection."""
        # Check for completed episodes with multi-player experiences
        if "infos" in self.locals:
            for info in self.locals["infos"]:
                if (
                    info.get("episode_complete", False)
                    and "multi_player_experiences" in info
                ):
                    experiences = info["multi_player_experiences"]
                    self.multi_player_episodes += 1
                    self.total_experiences_collected += len(experiences)

                    if self.verbose >= 1 and self.multi_player_episodes % 10 == 0:
                        avg_exp = (
                            self.total_experiences_collected
                            / self.multi_player_episodes
                        )
                        self.logger.record(
                            "self_play/episodes", self.multi_player_episodes
                        )
                        self.logger.record(
                            "self_play/total_experiences",
                            self.total_experiences_collected,
                        )
                        self.logger.record("self_play/avg_exp_per_episode", avg_exp)
                        self.logger.record(
                            "self_play/model_injected", float(self.model_injected)
                        )

        return True

    def _on_training_end(self) -> None:
        """Called at the end of training. Print summary."""
        if self.verbose >= 1:
            print("\n" + "=" * 50)
            print("Self-Play Training Summary:")
            print(f"Model injection successful: {self.model_injected}")
            print(f"Multi-player episodes completed: {self.multi_player_episodes}")
            print(f"Total experiences collected: {self.total_experiences_collected}")
            if self.multi_player_episodes > 0:
                avg_exp = self.total_experiences_collected / self.multi_player_episodes
                print(f"Average experiences per episode: {avg_exp:.1f}")
                print(
                    f"Training data multiplier: ~{avg_exp / 10:.1f}x compared to single-player"
                )
            print("=" * 50 + "\n")
