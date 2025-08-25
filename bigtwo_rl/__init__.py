"""Big Two RL Agent Library

A reinforcement learning library for training and evaluating AI agents that play Big Two.

Main components:
- bigtwo_rl.core: Game environment and RL wrapper (now with fixed 1,365-action space)
- bigtwo_rl.agents: Various agent implementations (fixed action space compatible)
- bigtwo_rl.training: Training infrastructure with configurable rewards/hyperparams
- bigtwo_rl.evaluation: Tournament system and evaluation tools
"""

__version__ = "0.2.0"  # Updated for fixed action space

# Configuration-based imports
from .config import create_wrapper, create_trainer, is_using_fixed_actions

# Core components - use configuration to determine which to import
if is_using_fixed_actions():
    # Fixed action space system (default)
    from .core.fixed_action_wrapper import FixedActionBigTwoWrapper as BigTwoRLWrapper
    from .agents.fixed_action_ppo_agent import FixedActionPPOAgent as PPOAgent
    from .agents.fixed_action_random_agent import FixedActionRandomAgent as RandomAgent
    from .agents.fixed_action_greedy_agent import FixedActionGreedyAgent as GreedyAgent
    from .training.fixed_action_trainer import FixedActionTrainer as Trainer
    from .evaluation.fixed_action_evaluator import FixedActionEvaluator as Evaluator
else:
    # Legacy system (backward compatibility)
    from .core.rl_wrapper import BigTwoRLWrapper
    from .agents.ppo_agent import PPOAgent
    from .agents.random_agent import RandomAgent
    from .agents.greedy_agent import GreedyAgent
    from .training.trainer import Trainer
    from .evaluation.evaluator import Evaluator

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

# Fixed action space specific exports
from .core.action_space import BigTwoActionSpace, HandType
from .core.action_system import BigTwoActionSystem

# Factory functions for configuration-aware creation
def create_environment(**kwargs):
    """Create Big Two environment using current configuration."""
    return create_wrapper(**kwargs)

def create_training_system(**kwargs):
    """Create trainer using current configuration."""
    return create_trainer(**kwargs)

# Training API exports - import these explicitly for training
# from bigtwo_rl.training import Trainer
# from bigtwo_rl.training.rewards import DefaultReward, SparseReward, etc.
# from bigtwo_rl.training.hyperparams import DefaultConfig, AggressiveConfig, etc.

__all__ = [
    # Core agents and components
    "BaseAgent",
    "BigTwoRLWrapper", 
    "GreedyAgent",
    "PPOAgent",
    "RandomAgent",
    "ToyBigTwoFullRules",
    "Trainer",
    "Evaluator",
    
    # Observation system
    "ObservationBuilder",
    "ObservationConfig",
    "minimal_observation",
    "standard_observation", 
    "memory_enhanced_observation",
    "strategic_observation",
    
    # Fixed action space components
    "BigTwoActionSpace",
    "BigTwoActionSystem",
    "HandType",
    
    # Factory functions
    "create_environment",
    "create_training_system",
    "create_wrapper",
    "create_trainer",
    
    # Configuration
    "is_using_fixed_actions",
]
