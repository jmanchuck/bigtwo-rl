# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Big Two RL Agent Library - A comprehensive reinforcement learning library for training AI agents to play Big Two (Chinese card game) using PPO (Proximal Policy Optimization). The library provides a clean, extensible API to experiment with different training approaches.

**Key Features**: 
- **High-Performance Core**: Fully vectorized numpy implementation (5-20x speedup)
- **Memory Optimized**: Boolean array representation (50-75% memory reduction)
- **Library Architecture**: Proper Python package with clear module organization
- **Extensible Training**: Configurable hyperparameters and custom reward functions 
- **Agent System**: Modular agent implementations (Random, Greedy, PPO)
- **Tournament Framework**: 4-player series and tournaments with statistics
- **Easy Integration**: Simple API for common workflows

## Development Commands

```bash
# Environment setup
uv sync                                    # Install dependencies from requirements.txt

# Library Usage (Primary Methods)
uv run python examples/train_agent.py               # Train agent with simple API
uv run python examples/evaluate_agent.py MODEL     # Evaluate trained model (4-player series)
uv run python examples/tournament_example.py       # Run 4-player tournament between agents
uv run python examples/custom_reward_example.py    # Train with custom reward function

# Testing and validation
uv run python tests/test_wrapper.py        # Test environment wrapper functionality
uv run python tests/test_cards.py          # Test card utilities (with numpy array support)
uv run python tests/test_rewards.py        # Test reward structure
uv run python tests/test_training.py       # Test training setup
uv run python tests/test_numpy_performance.py  # Performance benchmarks
uv run python tests/test_optimization_impact.py  # Optimization validation

# Interactive play
uv run python examples/play_vs_agent.py MODEL      # Play against trained agent

# Monitoring
uv run python -m tensorboard.main --logdir=./logs  # View training metrics (http://localhost:6006)
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
│   ├── evaluator.py                # Evaluator class: 4-player series assessment
│   └── tournament.py               # 4-player tournament system for agent competitions
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
- `Evaluator`: High-level model evaluation via 4-player series (agent + 3 opponents)
- `Tournament`: 4-player tournaments (round-robin over 4-agent tables)
- Head-to-head matchups are not supported (Big Two is strictly 4-player in this project)

**Game Environment (`bigtwo_rl.core`)**:
- `ToyBigTwoFullRules`: Complete Big Two game implementation with numpy vectorization
- `BigTwoRLWrapper`: Gymnasium-compatible RL environment with optimized observations
- 109-feature observation space with direct numpy array operations
- Dynamic action space with proper action masking
- **Performance**: Vectorized legal move generation, hand type identification with LRU cache

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

### Tournament Between Agents (4-player tables)
```python
from bigtwo_rl.agents import RandomAgent, GreedyAgent, PPOAgent
from bigtwo_rl.evaluation import Tournament

agents = [
    RandomAgent("Random-1"),
    RandomAgent("Random-2"),
    GreedyAgent("Greedy"),
    PPOAgent("./models/my_model/best_model", "MyAgent")
]

tournament = Tournament(agents)
results = tournament.run_round_robin(num_games=100)
print(results["tournament_summary"])
```

### Model Evaluation (4-player series)
```python
from bigtwo_rl.evaluation import Evaluator

evaluator = Evaluator(num_games=100)
results = evaluator.evaluate_model("./models/my_model/best_model")
print(results)
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

## Hyperparameters Explained (Big Two Context)

Understanding hyperparameters in terms of actual Big Two gameplay:

### Game Structure Hierarchy
1. **Move/Step**: Single card play action (play 3♦, pass, etc.)
2. **Game**: Complete Big Two game (deal cards → play until someone wins)
3. **Episode**: Multiple games grouped together (default: 5-10 games)
4. **Training Run**: Many episodes until total_timesteps reached

### Key Hyperparameters in Big Two Terms

**Episode Structure:**
- `games_per_episode` (5): How many complete Big Two games before the AI gets feedback
  - Why multiple games? Card dealing is random, so one game isn't enough signal
  - Agent only gets reward at episode end (after all 5 games)

**Data Collection:**
- `n_steps` (512): How many individual card plays to collect before updating the AI
  - In Big Two terms: ~10-25 complete games worth of moves (games are ~20-40 moves)
- `n_envs` (8): How many parallel Big Two tables running simultaneously
  - Like having 8 different card tables generating training data at once

**Learning Updates:**
- `batch_size` (64): How many individual moves to review together when updating strategy
- `n_epochs` (10): How many times to review the same batch of moves
  - Like replaying the same hands 10 times to learn from them

**Training Volume:**
- `total_timesteps` (25,000): Total individual card plays for entire training
  - At ~30 moves/game and 5 games/episode = ~150 moves/episode
  - So 25k timesteps ≈ 167 episodes ≈ 835 total games

**Example Training Session (default settings):**
- Train for 25,000 card plays total
- Collect 512 moves at a time from 8 parallel tables
- Every 512 moves, update the AI by reviewing batches of 64 moves, 10 times each
- Each training episode = 5 complete Big Two games before reward
- **Result**: ~835 total games played during training (why 30 seconds feels short!)


### Recommended hyperparams to try out

  n_steps Reduction:
  - Target: n_steps = ~150 (3-5 games worth)
  - Allows learning updates after every few complete games
  - Maintains some stability while increasing feedback frequency

  Batch Size Heuristics:
  - batch_size = 32-48 moves (~1-2 complete games)
  - Ensures each batch contains coherent game sequences
  - Avoids mixing early-game and late-game moves randomly

  Game-Aware Training:
  - games_per_episode = 3  # Smaller episodes, more frequent rewards
  - n_steps = 180         # ~6 games before update  

## Performance Optimizations (Updated: 2025-08-19)

### Numpy Vectorization Architecture
The library has been fully optimized with numpy vectorization for maximum performance:

**Core Data Structures**:
- **Hands**: `np.array(shape=(4,52), dtype=bool)` - 50% memory reduction
- **Game State**: All internal operations use numpy arrays
- **Observations**: Direct numpy operations, no intermediate conversions

**Vectorized Operations**:
- **Legal Move Generation**: Fully vectorized using `np.unique()`, `np.bincount()`
- **Hand Type Identification**: Vectorized with `np.diff()`, `np.array_equal()` + LRU cache
- **Card Operations**: Bitwise operations on boolean arrays
- **Observation Construction**: Direct numpy slicing and concatenation

**Performance Metrics**:
- **Environment Reset**: 0.044ms (22,775 resets/sec)
- **Game Steps**: 0.118ms (8,475 steps/sec)
- **Legal Moves**: 0.038ms (26,040 calls/sec)
- **Hand Identification**: Near-instant with 99.9%+ cache hit rate

**Memory Efficiency**:
- 50-75% reduction in memory usage vs previous list-based implementation
- Boolean arrays optimize both space and cache locality
- Direct numpy operations eliminate temporary allocations

### Interface Compatibility
- **Internal**: 100% numpy vectorized for maximum performance
- **External**: Clean conversion functions in `card_utils.py` for human interaction
- **Backward Compatibility**: All existing APIs work unchanged

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
- **High-Performance Training**: 8,000+ steps/sec enables rapid experimentation

## Development Notes

- Uses `uv` for Python dependency management
- Proper Python packaging with `pyproject.toml`
- Stable-Baselines3 for PPO implementation
- Tensorboard for training visualization
- Modular architecture enables easy experimentation
- Comprehensive example scripts for common workflows
- **Performance-First Design**: All core operations fully vectorized with numpy
- **Memory Optimized**: Boolean arrays and efficient data structures throughout