"""Training infrastructure for Big Two agents."""

from .trainer import Trainer
from .multi_player_ppo import MultiPlayerPPO
from .callbacks import BigTwoMetricsCallback

# Import hyperparameter configurations
from .hyperparams import (
    DefaultConfig,
    AggressiveConfig,
    ConservativeConfig,
    FastExperimentalConfig,
    OptimizedConfig,
    ReferenceExactConfig,
)

__all__ = [
    "Trainer",
    "MultiPlayerPPO",
    "BigTwoMetricsCallback",
    # Hyperparameter configurations
    "DefaultConfig",
    "AggressiveConfig",
    "ConservativeConfig",
    "FastExperimentalConfig",
    "OptimizedConfig",
    "ReferenceExactConfig",
]
