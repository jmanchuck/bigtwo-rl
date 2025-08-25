"""Training infrastructure for Big Two agents."""

from .callbacks import BigTwoMetricsCallback

# Import hyperparameter configurations
from .hyperparams import (
    AggressiveConfig,
    ConservativeConfig,
    DefaultConfig,
    FastExperimentalConfig,
    ReferenceExactConfig,
)
from .multi_player_buffer_enhanced import MultiPlayerRolloutBuffer
from .multi_player_ppo import MultiPlayerPPO
from .trainer import Trainer

__all__ = [
    "AggressiveConfig",
    "BigTwoMetricsCallback",
    "ConservativeConfig",
    # Hyperparameter configurations
    "DefaultConfig",
    "FastExperimentalConfig",
    "MultiPlayerPPO",
    "MultiPlayerRolloutBuffer",
    "ReferenceExactConfig",
    "Trainer",
]
