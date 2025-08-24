"""Training callbacks for Big Two agents."""

import traceback
from typing import Dict, Any
from stable_baselines3.common.callbacks import BaseCallback


class BigTwoMetricsCallback(BaseCallback):
    """
    Callback to log Big Two-specific training metrics to TensorBoard.

    Automatically detects episode completions and logs game performance,
    strategy usage, and opponent comparison metrics.
    """

    def __init__(self, verbose: int = 0):
        """
        Args:
            verbose: Verbosity level (0 = quiet, 1 = info)
        """
        super().__init__(verbose)

    def _on_step(self) -> bool:
        """Called after each step. Logs metrics when episodes complete."""
        # Check if episode just completed - handle both vectorized and single env
        dones = self.locals.get("dones")
        if dones is None:
            dones = self.locals.get("done")
        if dones is not None:
            # Handle vectorized env (list/array of dones): log if ANY env finished
            if hasattr(dones, "__len__") and len(dones) > 0:
                try:
                    episode_done = any(bool(d) for d in dones)
                except Exception:
                    episode_done = any([bool(d) for d in list(dones)])
            else:
                episode_done = bool(dones)
        else:
            # Fallback: check infos for episode end signal
            infos = self.locals.get("infos")
            if infos is None:
                infos = self.locals.get("info")
            if infos is None:
                infos = []
            episode_done = any(info.get("episode") is not None for info in infos)

        if episode_done:
            self._log_episode_metrics()
        return True

    def _log_episode_metrics(self) -> None:
        """Extract and log Big Two metrics from episode info."""
        try:
            # Extract metrics from info dict (works with vectorized environments)
            infos = self.locals.get("infos")
            if infos is None:
                infos = self.locals.get("info")
            if infos is None:
                infos = []

            # Aggregate Big Two metrics across any envs that finished this step
            metric_lists = {}
            for info in infos:
                # Each info corresponds to one sub-env; may or may not include metrics
                sub_metrics = {
                    k: v
                    for k, v in info.items()
                    if isinstance(k, str) and k.startswith("bigtwo/")
                }
                if not sub_metrics:
                    continue
                for k, v in sub_metrics.items():
                    metric_lists.setdefault(k, []).append(v)

            if metric_lists:
                # Compute mean across envs that reported this metric
                aggregated = {
                    k: (sum(vals) / len(vals))
                    for k, vals in metric_lists.items()
                    if len(vals) > 0
                }
                # Robust win-rate recompute if counters present
                gw = aggregated.get("bigtwo/games_won")
                gp = aggregated.get("bigtwo/games_completed")
                if gp is not None and gp > 0 and gw is not None:
                    aggregated["bigtwo/win_rate"] = float(gw) / float(gp)
                # Log aggregated metrics to TensorBoard
                for metric_name, value in aggregated.items():
                    self.logger.record(metric_name, value)
            else:
                # No Big Two metrics found in episode info
                pass

        except Exception as e:
            # Failed to log Big Two metrics
            traceback.print_exc()
