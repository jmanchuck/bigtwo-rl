"""Big Two RL Agent Library

A reinforcement learning library for training and evaluating AI agents that play Big Two.

Main components:
- bigtwo_rl.core: Game environment and RL wrapper
- bigtwo_rl.agents: Various agent implementations
- bigtwo_rl.training: Training infrastructure with configurable rewards/hyperparams
- bigtwo_rl.evaluation: Tournament system and evaluation tools
"""

__version__ = "0.1.0"

# Main API exports
from .agents import BaseAgent, GreedyAgent, PPOAgent, RandomAgent
from .core import BigTwoRLWrapper, ToyBigTwoFullRules
from .core.observation_builder import (
    ObservationBuilder,
    ObservationConfig,
    strategic_observation,
)

# Training API exports - import these explicitly for training
# from bigtwo_rl.training import Trainer
# from bigtwo_rl.training.rewards import DefaultReward, SparseReward, etc.
# from bigtwo_rl.training.hyperparams import DefaultConfig, AggressiveConfig, etc.

__all__ = [
    "BaseAgent",
    "BigTwoRLWrapper",
    "GreedyAgent",
    "ObservationBuilder",
    "ObservationConfig",
    "PPOAgent",
    "RandomAgent",
    "ToyBigTwoFullRules",
    "strategic_observation",
]
