"""Multi-player rollout buffer with explicit per-env player tracking."""

from collections import deque

import numpy as np
import torch as th
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer


class MultiPlayerRolloutBuffer(MaskableRolloutBuffer):
    """Rollout buffer for turn-based multi-player PPO."""

    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: th.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.n_envs = n_envs
        self.player_who_moved = None
        self.last_player_positions = [{i: deque(maxlen=4) for i in range(4)} for _ in range(n_envs)]
        self.immediate_rewards_assigned = 0
        self.games_completed = 0
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        current_player: int | np.ndarray | None = None,
        action_masks: np.ndarray | None = None,
    ) -> None:
        """Add one rollout step and optionally back-assign terminal rewards."""
        current_pos = self.pos
        players = self._normalize_current_players(current_player, current_pos)
        super().add(obs, action, reward, episode_start, value, log_prob, action_masks=action_masks)
        self.player_who_moved[current_pos, :] = players
        for env_idx, player_idx in enumerate(players):
            self.last_player_positions[env_idx][int(player_idx)].append(current_pos)

        for env_idx in range(self.n_envs):
            env_reward = reward[env_idx] if hasattr(reward, "__len__") and len(reward) > env_idx else reward
            if isinstance(env_reward, (list, np.ndarray)) and len(env_reward) == 4:
                self._assign_final_game_rewards_immediately(env_reward, env_idx)
                self.games_completed += 1

    def get_carryover(self, n_transitions: int = 4) -> dict[str, np.ndarray] | None:
        """Return the latest transitions to carry into the next rollout.

        Carrying turn-contiguous transitions across rollout boundaries improves
        terminal back-assignment and turn-based GAE continuity.
        """
        size = self.buffer_size if self.full else self.pos
        if size <= 0:
            return None

        n = min(int(n_transitions), size)
        if n <= 0:
            return None

        start = size - n
        sl = slice(start, size)
        carry = {
            "observations": np.array(self.observations[sl], copy=True),
            "actions": np.array(self.actions[sl], copy=True),
            "rewards": np.array(self.rewards[sl], copy=True),
            "episode_starts": np.array(self.episode_starts[sl], copy=True),
            "values": np.array(self.values[sl], copy=True),
            "log_probs": np.array(self.log_probs[sl], copy=True),
            "player_who_moved": np.array(self.player_who_moved[sl], copy=True),
        }
        if hasattr(self, "action_masks") and self.action_masks is not None:
            carry["action_masks"] = np.array(self.action_masks[sl], copy=True)
        return carry

    def prime_with_carryover(self, carryover: dict[str, np.ndarray] | None) -> int:
        """Pre-fill the beginning of a fresh rollout with carried transitions.

        Returns:
            Number of primed transitions.
        """
        if not carryover:
            return 0

        n = int(carryover["actions"].shape[0])
        if n <= 0:
            return 0
        if n >= self.buffer_size:
            n = self.buffer_size - 1
            if n <= 0:
                return 0

        self.observations[:n] = carryover["observations"][:n]
        self.actions[:n] = carryover["actions"][:n]
        self.rewards[:n] = carryover["rewards"][:n]
        self.episode_starts[:n] = carryover["episode_starts"][:n]
        self.values[:n] = carryover["values"][:n]
        self.log_probs[:n] = carryover["log_probs"][:n]
        self.player_who_moved[:n] = carryover["player_who_moved"][:n]
        if "action_masks" in carryover and hasattr(self, "action_masks") and self.action_masks is not None:
            self.action_masks[:n] = carryover["action_masks"][:n]

        self.pos = n
        self.full = (self.pos >= self.buffer_size)

        # Rebuild lightweight tracking structures for diagnostics.
        self.last_player_positions = [{i: deque(maxlen=4) for i in range(4)} for _ in range(self.n_envs)]
        for step_idx in range(n):
            for env_idx in range(self.n_envs):
                player_idx = int(self.player_who_moved[step_idx, env_idx])
                if 0 <= player_idx < 4:
                    self.last_player_positions[env_idx][player_idx].append(step_idx)
        return n

    def _normalize_current_players(
        self,
        current_player: int | np.ndarray | None,
        current_pos: int,
    ) -> np.ndarray:
        if current_player is None:
            import warnings

            warnings.warn(
                "current_player not provided to buffer.add(); using position fallback.",
                UserWarning,
            )
            return np.full(self.n_envs, current_pos % 4, dtype=int)

        if isinstance(current_player, np.ndarray):
            if current_player.size == 0:
                return np.zeros(self.n_envs, dtype=int)
            if current_player.size >= self.n_envs:
                return current_player[: self.n_envs].astype(int, copy=False)
            out = np.empty(self.n_envs, dtype=int)
            out[: current_player.size] = current_player.astype(int, copy=False)
            out[current_player.size :] = int(current_player[0])
            return out

        return np.full(self.n_envs, int(current_player), dtype=int)

    def _assign_final_game_rewards_immediately(self, game_rewards: list | np.ndarray, env_idx: int) -> None:
        """Assign terminal rewards to the last 4 turns in a specific env stream."""
        if len(game_rewards) != 4:
            return

        positions_to_assign = [((self.pos - i) % self.buffer_size) for i in range(1, 5) if self.pos >= i]
        while len(positions_to_assign) < 4:
            positions_to_assign.append(0)

        for i, buffer_pos in enumerate(positions_to_assign):
            player_who_moved = int(self.player_who_moved[buffer_pos, env_idx])
            if 0 <= player_who_moved < 4:
                self.rewards[buffer_pos, env_idx] = game_rewards[player_who_moved]
                self.immediate_rewards_assigned += 1
                if i > 0:
                    next_pos = (buffer_pos + 1) % self.buffer_size
                    self.episode_starts[next_pos, env_idx] = True
        for player_idx in range(4):
            self.last_player_positions[env_idx][player_idx].clear()

    def assign_final_game_rewards(self, game_rewards: list[float] | np.ndarray, env_idx: int = 0) -> None:
        """Public hook to assign terminal game rewards to the last 4 transitions.

        This is called from the custom PPO rollout collector when an env reports
        `info["final_rewards"]` at game end.
        """
        self._assign_final_game_rewards_immediately(game_rewards, env_idx)
        self.games_completed += 1

    def compute_multi_player_gae(self, gamma: float, gae_lambda: float) -> None:
        """Compute GAE per-player and per-env stream."""
        advantages = np.zeros_like(self.rewards)

        for env_idx in range(self.n_envs):
            for player_id in range(4):
                last_gae_lam = 0.0
                player_steps = [
                    step_idx for step_idx in range(self.buffer_size)
                    if self.player_who_moved[step_idx, env_idx] == player_id
                ]
                for i in reversed(range(len(player_steps))):
                    step_idx = player_steps[i]
                    if i + 1 < len(player_steps):
                        next_step_idx = player_steps[i + 1]
                        next_non_terminal = 1.0 - self.episode_starts[next_step_idx, env_idx]
                        next_values = self.values[next_step_idx, env_idx]
                    else:
                        next_non_terminal = 0.0
                        next_values = 0.0

                    delta = (
                        self.rewards[step_idx, env_idx]
                        + gamma * next_values * next_non_terminal
                        - self.values[step_idx, env_idx]
                    )
                    last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
                    advantages[step_idx, env_idx] = last_gae_lam

        self.advantages = advantages
        self.returns = advantages + self.values

    def get_statistics(self) -> dict:
        """Get buffer statistics for monitoring."""
        return {
            "immediate_rewards_assigned": self.immediate_rewards_assigned,
            "games_completed": self.games_completed,
            "player_position_tracking": [
                {f"player_{i}": len(positions[i]) for i in range(4)} for positions in self.last_player_positions
            ],
        }

    def reset(self) -> None:
        """Reset the buffer and clear player tracking/statistics."""
        super().reset()
        self.player_who_moved = np.zeros((self.buffer_size, self.n_envs), dtype=int)
        self.last_player_positions = [{i: deque(maxlen=4) for i in range(4)} for _ in range(self.n_envs)]
        self.immediate_rewards_assigned = 0
        self.games_completed = 0
