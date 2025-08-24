"""Hyperparameter configurations for different training experiments."""

import os
from dataclasses import dataclass
from abc import ABC


@dataclass(frozen=False)
class BaseConfig(ABC):
    """Base class for hyperparameter configurations."""

    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    games_per_episode: int
    n_envs: int

    def to_dict(self):
        """Convert to dictionary for backward compatibility."""
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "games_per_episode": self.games_per_episode,
            "n_envs": self.n_envs,
        }


@dataclass(frozen=False)
class DefaultConfig(BaseConfig):
    """Default hyperparameter configuration for balanced training."""

    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    games_per_episode: int = 5
    n_envs: int = min(8, max(2, (os.cpu_count() or 4) // 2))


@dataclass(frozen=False)
class AggressiveConfig(BaseConfig):
    """Aggressive hyperparameter configuration for faster, less stable training."""

    learning_rate: float = 1e-3
    n_steps: int = 256
    batch_size: int = 32
    n_epochs: int = 10
    gamma: float = 0.95
    gae_lambda: float = 0.9
    clip_range: float = 0.3
    games_per_episode: int = 3
    n_envs: int = min(6, max(2, (os.cpu_count() or 6) // 3))


@dataclass(frozen=False)
class ConservativeConfig(BaseConfig):
    """Conservative hyperparameter configuration for stable, slower training."""

    learning_rate: float = 1e-4
    n_steps: int = 1024
    batch_size: int = 128
    n_epochs: int = 15
    gamma: float = 0.995
    gae_lambda: float = 0.98
    clip_range: float = 0.1
    games_per_episode: int = 10
    n_envs: int = min(4, max(2, (os.cpu_count() or 4) // 2))


@dataclass(frozen=False)
class FastExperimentalConfig(BaseConfig):
    """Fast experimental hyperparameter configuration for quick testing."""

    learning_rate: float = 5e-4
    n_steps: int = 128
    batch_size: int = 16
    n_epochs: int = 3
    gamma: float = 0.9
    gae_lambda: float = 0.85
    clip_range: float = 0.25
    games_per_episode: int = 2
    n_envs: int = min(16, max(4, os.cpu_count() or 8))


@dataclass(frozen=False)
class ReferenceExactConfig(BaseConfig):
    """Exact hyperparameter match to reference implementation.
    
    This configuration replicates the exact settings from the reference
    Big Two PPO implementation to maximize compatibility and performance.
    Based on analysis of mainBig2PPOSimulation.py and network architecture.
    """
    
    learning_rate: float = 2.5e-4      # Exact match to reference (0.00025)
    n_steps: int = 20                  # Match reference batch collection
    batch_size: int = 64               # Match reference mini-batch size  
    n_epochs: int = 5                  # Match reference optimization epochs
    gamma: float = 0.995               # Exact match to reference
    gae_lambda: float = 0.95           # Exact match to reference
    clip_range: float = 0.2            # Match reference PPO clipping
    games_per_episode: int = 1         # Single game per episode like reference
    
    # Critical: match reference's vectorized environment setup
    n_envs: int = 16                   # Reference uses multiple parallel games
    
    def __post_init__(self):
        """Post-initialization to ensure proper batch structure."""
        # Ensure batch_size works with n_envs * n_steps
        # Reference uses mini-batches within the total collection
        total_batch = self.n_envs * self.n_steps  # 16 * 20 = 320
        
        # Adjust batch_size to be compatible
        if self.batch_size > total_batch:
            self.batch_size = total_batch // 4  # Use quarter of total as mini-batch
