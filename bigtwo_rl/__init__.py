"""Big Two RL Agent Library

A reinforcement learning library for training and evaluating AI agents that play Big Two.

Main components:
- bigtwo_rl.core: Game environment and RL wrapper (1,365-action space)
- bigtwo_rl.agents: Agent implementations (PPO, Random, Greedy baselines)
- bigtwo_rl.training: Training infrastructure with configurable rewards/hyperparams
- bigtwo_rl.evaluation: Tournament system and evaluation tools
"""

__version__ = "1.0.0"  # Clean single-system version

# Core components
from .core.bigtwo_wrapper import BigTwoWrapper
from .agents.ppo_agent import PPOAgent
from .agents.random_agent import RandomAgent
from .agents.greedy_agent import GreedyAgent
from .training.trainer import Trainer
from .evaluation.evaluator import Evaluator
from .evaluation.tournament import Tournament

# Always available components
from .agents.base_agent import BaseAgent
from .core.bigtwo import ToyBigTwoFullRules
from .core.observation_builder import (
    ObservationBuilder,
    ObservationConfig,
    minimal_observation,
    standard_observation,
    memory_enhanced_observation,
    strategic_observation,
)

# Action space components
from .core.action_space import BigTwoActionSpace, HandType
from .core.action_system import BigTwoActionSystem

# Backward compatibility aliases
BigTwoRLWrapper = BigTwoWrapper  # For existing code

__all__ = [
    # Core agents and components
    "BaseAgent",
    "BigTwoWrapper",
    "BigTwoRLWrapper",  # Backward compatibility
    "GreedyAgent",
    "PPOAgent",
    "RandomAgent",
    "ToyBigTwoFullRules",
    "Trainer",
    "Evaluator",
    "Tournament",
    
    # Observation system
    "ObservationBuilder",
    "ObservationConfig",
    "minimal_observation",
    "standard_observation", 
    "memory_enhanced_observation",
    "strategic_observation",
    
    # Action space components
    "BigTwoActionSpace",
    "BigTwoActionSystem",
    "HandType",
]