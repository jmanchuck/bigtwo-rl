# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Big Two RL Agent - A reinforcement learning experiment training an AI to play Big Two (Chinese card game) using PPO (Proximal Policy Optimization). The agent learns optimal strategies through self-play without explicit rules programming.

**Key Features**: Modular agent system, configurable hyperparameters, pluggable reward functions, and agent tournament system for comparing different training approaches.

## Development Commands

```bash
# Environment setup
uv sync                           # Install dependencies from requirements.txt

# Testing and validation
uv run test_wrapper.py            # Test environment wrapper functionality
uv run test_cards.py              # Test card utilities and hand detection
uv run test_rewards.py            # Test reward structure
uv run test_training.py           # Test training setup

# Configurable Training (NEW - Primary Method)
uv run train_with_config.py --list-configs          # List all available configurations
uv run train_with_config.py aggressive sparse 25000 # Train with aggressive hyperparams + sparse rewards
uv run train_with_config.py conservative default 50000 model_name  # Train with custom model name
uv run train_with_config.py fast_experimental progressive 10000    # Quick experimental training

# Agent Competition (NEW)
uv run python tournament.py example                 # Run example tournament with available agents
# Or use programmatically for custom tournaments

# Evaluation and testing
uv run evaluate.py ./models/best_model              # Benchmark against random/greedy baselines
uv run evaluate.py ./models/model_name/final_model  # Evaluate specific trained model
uv run play_vs_agent.py ./models/best_model         # Interactive CLI to play against trained agent

# Monitoring
tensorboard --logdir=./logs                         # View training metrics
tensorboard --logdir=./logs/model_name              # View specific model training
```

## Architecture

### Core Components

**Environment (`bigtwo.py`)**:
- `ToyBigTwoFullRules`: Full Big Two game implementation with all card combinations
- 279 lines implementing singles, pairs, trips, and 5-card hands (straights, flushes, etc.)
- Big Two card rankings: 2 highest (12), A second highest (11), down to 3 lowest (0)
- Supports 2-5 players with proper hand size distribution

**RL Wrapper (`rl_wrapper.py`)**:
- `BigTwoRLWrapper`: Gymnasium-compatible environment for Stable-Baselines3
- Fixed 57-dim observation space: hand_binary(52) + last_rank(1) + hand_sizes(4)
- Dynamic action space with action masking for illegal moves
- Multi-game episodes (default 10 games) for stable skill assessment

**Configurable Training (`train_with_config.py`)**:
- PPO with selectable hyperparameter configurations (default, aggressive, conservative, fast_experimental)
- Pluggable reward functions (default, sparse, aggressive_penalty, progressive, ranking)
- SubprocVecEnv for true multiprocessing (configurable parallel environments)
- EvalCallback for model checkpointing and progress tracking
- Tensorboard logging with organized model-specific directories

**Agent System (`agents/`)**:
- `BaseAgent`: Common interface for all agent types
- `RandomAgent`: Random baseline for evaluation
- `GreedyAgent`: Greedy baseline (lowest card preference)
- `PPOAgent`: Wrapper for trained PPO models

**Tournament System (`tournament.py`)**:
- Head-to-head agent matchups
- Round-robin tournaments with multiple agents
- Win rate statistics and performance comparisons
- Support for agent vs agent competitions

**Evaluation Suite**:
- `evaluate.py`: Benchmark trained models vs baselines using new agent system
- `play_vs_agent.py`: Interactive CLI with number-based move selection
- `card_utils.py`: Utility functions for card representation and sorting

### Key RL Concepts

**Observation Space (109 features)**:
```python
[hand_binary(52), last_play_binary(52), hand_sizes(4), last_play_exists(1)]
```

**Action Space**: Dynamic based on legal moves
- Singles: Play any card from hand
- Pairs: Play any valid pair
- Trips: Play any valid three-of-a-kind  
- 5-card hands: Straights, flushes, full house, four-of-a-kind, straight flush
- Pass: Available when not starting a trick

**Multi-Game Episodes**: Each training episode consists of multiple games (configurable) with reward only at episode end. This addresses card dealing randomness and focuses learning on true skill.

**Configurable Reward Functions**:
- `default`: Win +5, loss penalty scaled by remaining cards (original)
- `sparse`: Simple win (+1) vs loss (-1) 
- `aggressive_penalty`: Higher penalties for losing with many cards
- `progressive`: Rewards progress (fewer cards = better reward)
- `ranking`: Rewards based on final ranking among all players

Episode reward = average performance across all games in episode

## File Structure

```
bigtwo-agent/
├── configs/                    # Configuration management
│   ├── hyperparams.py         # 4 hyperparameter configurations
│   └── rewards.py             # 5 reward function implementations
├── agents/                     # Modular agent system
│   ├── base_agent.py          # Common agent interface
│   ├── random_agent.py        # Random baseline
│   ├── greedy_agent.py        # Greedy baseline  
│   └── ppo_agent.py           # PPO model wrapper
├── bigtwo.py                  # Complete Big Two game environment (279 lines)
├── rl_wrapper.py              # Gymnasium wrapper for RL training
├── train_with_config.py       # Configurable PPO training (primary method)
├── tournament.py              # Agent vs agent competition system
├── evaluate.py                # Model evaluation vs baselines
├── play_vs_agent.py           # Human vs agent CLI interface
├── card_utils.py              # Card utilities and display functions
├── test_*.py                  # Comprehensive test suite
├── models/                    # Saved model checkpoints organized by experiment
│   ├── best_model.zip         # Legacy best model
│   ├── final_model.zip        # Legacy final model
│   └── {experiment_name}/     # Organized model directories
└── logs/                      # Tensorboard logs organized by experiment
    └── {experiment_name}/     # Model-specific training logs
```

## Experimentation Workflow

### Hyperparameter Experiments
```bash
# List available configurations
uv run train_with_config.py --list-configs

# Train different hyperparameter sets
uv run train_with_config.py aggressive sparse 25000
uv run train_with_config.py conservative default 50000
uv run train_with_config.py fast_experimental progressive 10000
```

### Reward Function Experiments  
```bash
# Compare different reward functions with same hyperparams
uv run train_with_config.py default sparse 25000 sparse_experiment
uv run train_with_config.py default progressive 25000 progressive_experiment
uv run train_with_config.py default ranking 25000 ranking_experiment
```

### Agent Competition
```bash
# Run tournaments between different models
python -c "
from tournament import run_round_robin_tournament
from agents import RandomAgent, GreedyAgent, PPOAgent

agents = [
    RandomAgent('Random'),
    GreedyAgent('Greedy'), 
    PPOAgent('./models/sparse_experiment/best_model', 'Sparse-PPO'),
    PPOAgent('./models/progressive_experiment/best_model', 'Progressive-PPO')
]

results = run_round_robin_tournament(agents, num_games=100)
print(results['tournament_summary'])
"
```

## Training Results

Current implementation achieves:
- 100% win rate vs random and greedy baselines after 50k timesteps
- Episode length reduction from 69→43 steps (learned efficiency) 
- Successful convergence with full Big Two complexity (all hand types)
- Modular system enables easy comparison of different training approaches

## Development Notes

- Uses `uv` for Python dependency management
- No pyproject.toml - dependencies in requirements.txt
- Stable-Baselines3 for PPO implementation
- Tensorboard for training visualization
- Modular agent system for easy experimentation
- Organized model/log directories by experiment name