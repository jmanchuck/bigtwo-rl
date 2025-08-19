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
from .core import BigTwoRLWrapper, ToyBigTwoFullRules
from .core.observation_builder import ObservationBuilder, ObservationConfig
from .agents import BaseAgent, RandomAgent, GreedyAgent, PPOAgent

# Training API exports - import these explicitly for training
# from bigtwo_rl.training import Trainer
# from bigtwo_rl.training.rewards import DefaultReward, SparseReward, etc.
# from bigtwo_rl.training.hyperparams import DefaultConfig, AggressiveConfig, etc.

__all__ = [
    "BigTwoRLWrapper",
    "ToyBigTwoFullRules",
    "ObservationBuilder",
    "ObservationConfig",
    "BaseAgent",
    "RandomAgent",
    "GreedyAgent",
    "PPOAgent",
]
