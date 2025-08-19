"""Hyperparameter configurations for different training experiments."""

import os
from typing import Dict, Any, List

CONFIGS = {
    "default": {
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "games_per_episode": 5,
        "n_envs": min(8, max(2, (os.cpu_count() or 4) // 2)),
    },
    "aggressive": {
        "learning_rate": 1e-3,
        "n_steps": 256,
        "batch_size": 32,
        "n_epochs": 5,
        "gamma": 0.95,
        "gae_lambda": 0.9,
        "clip_range": 0.3,
        "games_per_episode": 3,
        "n_envs": min(6, max(2, (os.cpu_count() or 6) // 3)),
    },
    "conservative": {
        "learning_rate": 1e-4,
        "n_steps": 1024,
        "batch_size": 128,
        "n_epochs": 15,
        "gamma": 0.995,
        "gae_lambda": 0.98,
        "clip_range": 0.1,
        "games_per_episode": 10,
        "n_envs": min(4, max(2, (os.cpu_count() or 4) // 2)),
    },
    "fast_experimental": {
        "learning_rate": 5e-4,
        "n_steps": 128,
        "batch_size": 16,
        "n_epochs": 3,
        "gamma": 0.9,
        "gae_lambda": 0.85,
        "clip_range": 0.25,
        "games_per_episode": 2,
        "n_envs": min(16, max(4, os.cpu_count() or 8)),
    },
}


def get_config(name: str = "default") -> Dict[str, Any]:
    """Get hyperparameter configuration by name."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name].copy()


def list_configs() -> List[str]:
    """List all available configuration names."""
    return list(CONFIGS.keys())
