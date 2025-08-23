"""Training infrastructure for Big Two agents."""

from .trainer import Trainer
from .opponent_pool import OpponentPool
from .callbacks import BigTwoMetricsCallback

# Import hyperparameter configurations
from .hyperparams import (
    DefaultConfig,
    AggressiveConfig,
    ConservativeConfig,
    FastExperimentalConfig,
    OptimizedConfig,
)

# Import all reward functions from the rewards module
from .rewards import (
    BaseReward,
    MoveQualityReward,
    DefaultReward,
    SparseReward,
    AggressivePenaltyReward,
    ProgressiveReward,
    RankingReward,
    ScoreMarginReward,
    StrategicReward,
    ComplexMoveReward,
    ZeroSumReward,
)

__all__ = [
    "Trainer",
    "OpponentPool",
    "BigTwoMetricsCallback",
    # Hyperparameter configurations
    "DefaultConfig",
    "AggressiveConfig",
    "ConservativeConfig", 
    "FastExperimentalConfig",
    "OptimizedConfig",
    # Reward functions
    "BaseReward",
    "MoveQualityReward",
    "DefaultReward",
    "SparseReward",
    "AggressivePenaltyReward",
    "ProgressiveReward",
    "RankingReward",
    "ScoreMarginReward",
    "StrategicReward",
    "ComplexMoveReward",
    "ZeroSumReward",
]
