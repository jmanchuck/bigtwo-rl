"""Opponent pool and provider for mixing snapshots and heuristic opponents.

This module exposes two entry points:
 - OpponentPool: manages snapshot discovery and sampling.
 - EnvOpponentProvider: thin wrapper that BigTwoRLWrapper can call per episode.
"""

from __future__ import annotations

import glob
import os
import random
from typing import Dict, List, Optional

from ..agents import GreedyAgent, RandomAgent, PPOAgent


class OpponentPool:
    """Manages a pool of frozen PPO checkpoints plus heuristic opponents.

    Parameters:
        snapshot_dir: directory where PPO checkpoints are saved (e.g., models/run_name)
        mixture: dict with weights for {"snapshots", "greedy", "random"}
        max_cached: max number of PPOAgent models to keep in-memory
    """

    def __init__(
        self,
        snapshot_dir: str,
        mixture: Optional[Dict[str, float]] = None,
        max_cached: int = 8,
    ):
        self.snapshot_dir = snapshot_dir
        self.mixture = mixture or {"snapshots": 0.6, "greedy": 0.3, "random": 0.1}
        self.max_cached = max_cached

        self._cached_agents: Dict[str, PPOAgent] = {}

    def _list_snapshot_paths(self) -> List[str]:
        patterns = [
            os.path.join(self.snapshot_dir, "**", "best_model.zip"),
            os.path.join(self.snapshot_dir, "**", "final_model.zip"),
            os.path.join(self.snapshot_dir, "**", "step_*", "model.zip"),
        ]
        paths: List[str] = []
        for pat in patterns:
            paths.extend(glob.glob(pat, recursive=True))
        # Unique, sorted for determinism
        paths = sorted(list(set(paths)))
        return paths

    def _sample_category(self) -> str:
        cats = ["snapshots", "greedy", "random"]
        weights = [self.mixture.get(k, 0.0) for k in cats]
        s = sum(weights)
        if s <= 0:
            # default to snapshots then greedy, then random
            weights = [0.6, 0.3, 0.1]
            s = 1.0
        weights = [w / s for w in weights]
        r = random.random()
        cum = 0.0
        for k, w in zip(cats, weights):
            cum += w
            if r <= cum:
                return k
        return cats[-1]

    def _get_snapshot_agent(self) -> Optional[PPOAgent]:
        paths = self._list_snapshot_paths()
        if not paths:
            return None
        path = random.choice(paths)
        if path in self._cached_agents:
            return self._cached_agents[path]
        # Maintain small cache
        if len(self._cached_agents) >= self.max_cached:
            # evict random
            k = random.choice(list(self._cached_agents.keys()))
            del self._cached_agents[k]
        try:
            agent = PPOAgent(model_path=path)
            self._cached_agents[path] = agent
            return agent
        except Exception:
            return None

    def sample_opponent(self):
        cat = self._sample_category()
        if cat == "greedy":
            return GreedyAgent("Greedy")
        if cat == "random":
            return RandomAgent("Random")
        # snapshots
        agent = self._get_snapshot_agent()
        if agent is not None:
            return agent
        # fallback order
        return GreedyAgent("Greedy")


class EnvOpponentProvider:
    """Adapter used by BigTwoRLWrapper to obtain per-episode opponents.

    It samples one opponent per non-controlled seat using the OpponentPool.
    """

    def __init__(self, pool: OpponentPool):
        self.pool = pool

    def get_episode_opponents(
        self, num_players: int, controlled_player: int
    ) -> Dict[int, object]:
        opponents: Dict[int, object] = {}
        for p in range(num_players):
            if p == controlled_player:
                continue
            opponents[p] = self.pool.sample_opponent()
        return opponents
