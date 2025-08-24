"""Training infrastructure for Big Two agents."""

from .trainer import Trainer
from .callbacks import BigTwoMetricsCallback

# Import hyperparameter configurations
from .hyperparams import (
    DefaultConfig,
    AggressiveConfig,
    ConservativeConfig,
    FastExperimentalConfig,
    OptimizedConfig,
)

__all__ = [
    "Trainer",
    "BigTwoMetricsCallback",
    # Hyperparameter configurations
    "DefaultConfig",
    "AggressiveConfig",
    "ConservativeConfig",
    "FastExperimentalConfig",
    "OptimizedConfig",
]
