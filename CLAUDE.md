# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Big Two RL Agent Library - A comprehensive reinforcement learning library for training AI agents to play Big Two (Chinese card game) using PPO (Proximal Policy Optimization). The library provides a clean, extensible API to experiment with different training approaches.

**Key Features**: 
- **Library Architecture**: Proper Python package with clear module organization
- **Extensible Training**: Configurable hyperparameters and custom reward functions 
- **Agent System**: Modular agent implementations (Random, Greedy, PPO)
- **Tournament Framework**: Agent vs agent competitions with statistics
- **Easy Integration**: Simple API for common workflows

## Development Commands

```bash
# Environment setup
uv sync                                    # Install dependencies from requirements.txt

# Library Usage (Primary Methods)
uv run python examples/train_agent.py               # Train agent with simple API
uv run python examples/evaluate_agent.py MODEL     # Evaluate trained model
uv run python examples/tournament_example.py       # Run tournament between agents
uv run python examples/custom_reward_example.py    # Train with custom reward function

# Testing and validation
uv run python tests/test_wrapper.py        # Test environment wrapper functionality
uv run python tests/test_cards.py          # Test card utilities and hand detection
uv run python tests/test_rewards.py        # Test reward structure
uv run python tests/test_training.py       # Test training setup

# Interactive play
uv run python examples/play_vs_agent.py MODEL      # Play against trained agent

# Monitoring
tensorboard --logdir=./logs                        # View training metrics
```

## Architecture

### Library Structure

```
bigtwo_rl/                           # Main library package
├── __init__.py                      # Main exports: BigTwoRLWrapper, agents
├── core/                            # Core game components
│   ├── bigtwo.py                   # Complete Big Two game implementation
│   ├── rl_wrapper.py               # Gymnasium-compatible RL wrapper
│   └── card_utils.py               # Card utilities and display functions
├── agents/                          # Agent implementations
│   ├── base_agent.py               # Common agent interface
│   ├── random_agent.py             # Random baseline
│   ├── greedy_agent.py             # Greedy baseline  
│   └── ppo_agent.py                # PPO model wrapper
├── training/                        # Training infrastructure
│   ├── trainer.py                  # Main Trainer class
│   ├── hyperparams.py              # Hyperparameter configurations
│   └── rewards.py                  # Reward functions + BaseReward class
├── evaluation/                      # Evaluation and competition
│   ├── evaluator.py                # Evaluator class for model assessment
│   └── tournament.py               # Tournament system for agent competitions
└── utils/                           # Utilities and helpers

examples/                            # Clear usage examples
├── train_agent.py                  # Simple training example
├── evaluate_agent.py               # Model evaluation example
├── tournament_example.py           # Tournament setup example
├── custom_reward_example.py        # Custom reward function example
└── play_vs_agent.py                # Interactive play vs agent

tests/                               # Comprehensive test suite
models/                              # Saved model checkpoints
logs/                                # Tensorboard training logs
```

### Core Components

**Training System (`bigtwo_rl.training`)**:
- `Trainer`: High-level training class with configurable rewards/hyperparams
- `BaseReward`: Abstract class for custom reward functions
- Built-in reward functions: default, sparse, aggressive_penalty, progressive, ranking
- Hyperparameter configurations: default, aggressive, conservative, fast_experimental

**Agent System (`bigtwo_rl.agents`)**:
- `BaseAgent`: Common interface for all agent types
- `RandomAgent`: Random baseline for evaluation
- `GreedyAgent`: Greedy baseline (lowest card preference)
- `PPOAgent`: Wrapper for trained PPO models

**Evaluation System (`bigtwo_rl.evaluation`)**:
- `Evaluator`: High-level model evaluation against baselines
- `Tournament`: Agent vs agent competitions with statistics
- Round-robin tournaments and head-to-head matchups

**Game Environment (`bigtwo_rl.core`)**:
- `ToyBigTwoFullRules`: Complete Big Two game implementation (279 lines)
- `BigTwoRLWrapper`: Gymnasium-compatible RL environment 
- 109-feature observation space with hand encoding and game state
- Dynamic action space with proper action masking

## Usage Examples

### Basic Training
```python
from bigtwo_rl.training import Trainer

# Simple training with defaults
trainer = Trainer()
model, model_dir = trainer.train(total_timesteps=25000)
```

### Custom Reward Function
```python
from bigtwo_rl.training import Trainer, BaseReward

class MyReward(BaseReward):
    def calculate(self, winner, player, cards_left, all_cards=None):
        if player == winner:
            return 10
        return -(cards_left ** 2) * 0.5

trainer = Trainer(reward_function=MyReward(), hyperparams="aggressive")
model, model_dir = trainer.train(total_timesteps=15000)
```

### Tournament Between Agents
```python
from bigtwo_rl.agents import RandomAgent, GreedyAgent, PPOAgent
from bigtwo_rl.evaluation import Tournament

agents = [
    RandomAgent("Random"),
    GreedyAgent("Greedy"),
    PPOAgent("./models/my_model/best_model", "MyAgent")
]

tournament = Tournament(agents)
results = tournament.run_round_robin(num_games=100)
print(results["tournament_summary"])
```

### Model Evaluation
```python
from bigtwo_rl.evaluation import Evaluator

evaluator = Evaluator(num_games=100)
results = evaluator.evaluate_model("./models/my_model/best_model")
print(results["tournament_summary"])
```

## Key RL Concepts

**Observation Space (109 features)**:
```python
[hand_binary(52), last_play_binary(52), hand_sizes(4), last_play_exists(1)]
```

**Action Space**: Dynamic based on legal moves
- Singles, pairs, trips, 5-card hands (straights, flushes, etc.)
- Pass action available when not starting a trick

**Multi-Game Episodes**: Each training episode consists of multiple games (configurable) with reward only at episode end to address card dealing randomness.

**Reward Functions**:
- `default`: Win +5, loss penalty scaled by remaining cards
- `sparse`: Simple win (+1) vs loss (-1) 
- `aggressive_penalty`: Higher penalties for losing with many cards
- `progressive`: Rewards progress (fewer cards = better reward)
- `ranking`: Rewards based on final ranking among all players

## Development Workflow

### Library Installation
```bash
# Development install
pip install -e .

# Use in other projects
pip install bigtwo-rl
```

### Adding Custom Agents
```python
from bigtwo_rl.agents import BaseAgent

class MyAgent(BaseAgent):
    def get_action(self, observation, action_mask=None):
        # Your agent logic here
        return action_index
    
    def reset(self):
        # Reset agent state
        pass
```

### Extending Reward Functions
```python
from bigtwo_rl.training.rewards import BaseReward

class MyReward(BaseReward):
    def calculate(self, winner_player, player_idx, cards_left, all_cards_left=None):
        # Your reward logic here
        return reward_value
```

## Training Results

Current implementation achieves:
- 100% win rate vs random and greedy baselines after 50k timesteps
- Episode length reduction from 69→43 steps (learned efficiency) 
- Successful convergence with full Big Two complexity (all hand types)
- Modular system enables easy comparison of different training approaches

## Development Notes

- Uses `uv` for Python dependency management
- Proper Python packaging with `pyproject.toml`
- Stable-Baselines3 for PPO implementation
- Tensorboard for training visualization
- Modular architecture enables easy experimentation
- Comprehensive example scripts for common workflows