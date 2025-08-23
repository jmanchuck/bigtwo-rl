# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Big Two RL Agent Library - A comprehensive reinforcement learning library for training AI agents to play Big Two (Chinese card game) using PPO (Proximal Policy Optimization). The library provides a clean, extensible API to experiment with different training approaches.

**Key Features**: 
- **High-Performance Core**: Fully vectorized numpy implementation (5-20x speedup)
- **Memory Optimized**: Boolean array representation (50-75% memory reduction)
- **Library Architecture**: Proper Python package with clear module organization
- **Extensible Training**: 8 reward functions + 4 hyperparameter configurations
- **Agent System**: Modular agent implementations (Random, Greedy, PPO)
- **Tournament Framework**: 4-player series and tournaments with multiprocessing
- **Configurable Observations**: 15+ observation features for experimentation

## Development Commands

```bash
# Environment setup
uv sync                                    # Install dependencies from requirements.txt

# Testing and validation
uv run python tests/test_wrapper.py        # Test environment wrapper functionality
uv run python tests/test_cards.py          # Test card utilities (with numpy array support)
uv run python tests/test_rewards.py        # Test reward structure
uv run python tests/test_training.py       # Test training setup
uv run python tests/test_numpy_performance.py  # Performance benchmarks
uv run python tests/test_optimization_impact.py  # Optimization validation

# Interactive play (existing examples)
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

examples/                            # Example scripts
├── play_vs_agent.py                # Interactive play vs agent

tests/                               # Comprehensive test suite
models/                              # Saved model checkpoints
logs/                                # Tensorboard training logs
```

### Core Components

**Training System (`bigtwo_rl.training`)**:
- `Trainer`: High-level training class with configurable rewards/hyperparams/observations
- `BaseReward`: Abstract class for custom reward functions with move bonuses
- **8 Built-in Reward Functions**:
  - `DefaultReward`: Balanced win/loss with card-based penalties
  - `SparseReward`: Simple win/loss only  
  - `AggressivePenaltyReward`: Higher penalties for many cards
  - `ProgressiveReward`: Rewards progress (fewer cards)
  - `RankingReward`: Rewards based on final ranking
  - `ScoreMarginReward`: Continuous reward based on card advantage
  - `StrategicReward`: Advanced strategic play encouragement
  - `ComplexMoveReward`: Bonuses for complex card combinations
- **4 Hyperparameter Configurations**: `DefaultConfig`, `AggressiveConfig`, `ConservativeConfig`, `FastExperimentalConfig`

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
- **Configurable Observation Space**: 15+ features, from minimal (57) to strategic (300+ features)
- Dynamic action space with proper action masking
- **Performance**: Vectorized legal move generation, hand type identification with LRU cache

**Observation System (`bigtwo_rl.core.observation_builder`)**:
- `ObservationConfig`: Configure exactly what the agent observes (15+ feature types)
- Pre-built configs: `minimal_observation` (57), `standard_observation` (109), `memory_enhanced_observation` (217), `strategic_observation` (300+)
- Custom configs: Hand types, opponent modeling, strategic features, memory features

## Usage Examples

### Basic Training
```python
from bigtwo_rl.training import Trainer
from bigtwo_rl.training.rewards import DefaultReward
from bigtwo_rl.training.hyperparams import DefaultConfig
from bigtwo_rl import standard_observation

# Training with all required components
trainer = Trainer(
    reward_function=DefaultReward(),
    hyperparams=DefaultConfig(),
    observation_config=standard_observation()  # Required
)
model, model_dir = trainer.train(total_timesteps=25000)
```

### Custom Reward Function
```python
from bigtwo_rl.training import Trainer
from bigtwo_rl.training.rewards import BaseReward
from bigtwo_rl.training.hyperparams import AggressiveConfig
from bigtwo_rl import minimal_observation

class MyReward(BaseReward):
    def game_reward(self, winner_player, player_idx, cards_left, all_cards_left=None):
        if player_idx == winner_player:
            return 10
        return -(cards_left ** 2) * 0.5
    
    def episode_bonus(self, games_won, total_games, avg_cards_left):
        return 0  # No episode bonus

trainer = Trainer(
    reward_function=MyReward(), 
    hyperparams=AggressiveConfig(),
    observation_config=minimal_observation()  # Use minimal for faster training
)
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
results = tournament.run(num_games=100)
print(results["tournament_summary"])
```

### Model Evaluation (4-player series)
```python
from bigtwo_rl.evaluation import Evaluator

evaluator = Evaluator(num_games=100)
results = evaluator.evaluate_model("./models/my_model/best_model")
# Access results: win_rates, avg_cards_left, total wins, game history
```

### Advanced Training with Custom Observations
```python
from bigtwo_rl.training import Trainer
from bigtwo_rl.training.rewards import StrategicReward, ComplexMoveReward
from bigtwo_rl.training.hyperparams import ConservativeConfig
from bigtwo_rl.core.observation_builder import ObservationConfig

# Custom observation for strategic play
strategic_obs = ObservationConfig(
    include_hand=True,
    include_last_play=True,
    include_hand_sizes=True,
    include_played_cards=True,        # Memory of all played cards
    include_pass_history=True,        # Who passed on current trick
    include_power_cards_remaining=True # Track high-value cards (2s, Aces)
)

# Train agent to play complex hands with strategic awareness
trainer = Trainer(
    reward_function=ComplexMoveReward(five_card_bonus=0.2),  # Bonus for 5-card hands
    hyperparams=ConservativeConfig(),     # Stable training
    observation_config=strategic_obs      # Rich observations
)
model, model_dir = trainer.train(total_timesteps=50000)
```

### Available Reward Functions
```python
from bigtwo_rl.training.rewards import *

# Simple rewards
DefaultReward()           # Balanced win/loss with card penalties
SparseReward()           # Simple win/loss only

# Strategic rewards  
StrategicReward()        # Encourages sophisticated play patterns
ProgressiveReward()      # Rewards progress (fewer cards)
RankingReward()          # Rewards based on final ranking

# Advanced rewards
ScoreMarginReward()      # Continuous reward based on card advantage
ComplexMoveReward(five_card_bonus=0.1)  # Bonuses for complex combinations
AggressivePenaltyReward()  # Higher penalties for poor performance
```

## Key RL Concepts

**Configurable Observation Space**:
- **Minimal** (57 features): `[hand(52), hand_sizes(4), last_play_exists(1)]`
- **Standard** (109 features): `[hand(52), last_play(52), hand_sizes(4), last_play_exists(1)]`  
- **Memory Enhanced** (217 features): Adds played cards history and remaining deck
- **Strategic** (300+ features): Adds opponent modeling, power cards, trick history, play patterns

**Action Space**: Dynamic based on legal moves
- Singles, pairs, trips, 5-card hands (straights, flushes, etc.)
- Pass action available when not starting a trick
- Proper action masking prevents invalid moves

**Multi-Game Episodes**: Each training episode consists of multiple games (configurable) with rewards given after each game plus episode bonus to address card dealing randomness.

**Reward Structure**:
- **Game Rewards**: Immediate feedback after each game (win/loss + card penalties)
- **Episode Bonus**: Additional reward based on overall episode performance  
- **Move Bonuses**: Optional rewards for specific move types (5-card hands, pairs)

## Hyperparameters Explained (Big Two Context)

Understanding hyperparameters in terms of actual Big Two gameplay:

### Game Structure Hierarchy
1. **Move/Step**: Single card play action (play 3♦, pass, etc.)
2. **Game**: Complete Big Two game (deal cards → play until someone wins)
3. **Episode**: Multiple games grouped together (default: 5-10 games)
4. **Training Run**: Many episodes until total_timesteps reached

### Hyperparameter Categories & Explanations

**Core RL Parameters:**
- `learning_rate` (1e-4 to 1e-3): How quickly the neural network updates its weights
  - Higher = faster learning but less stable convergence
  - Lower = slower but more stable training
- `gamma` (0.9 to 0.995): Discount factor for future rewards
  - High values (0.99+) = agent values long-term strategy 
  - Low values (0.9) = agent focuses on immediate rewards
  - In Big Two: higher gamma encourages winning games vs just playing cards
- `gae_lambda` (0.85 to 0.98): Generalized Advantage Estimation parameter for variance/bias tradeoff
  - High values = lower bias but higher variance in advantage estimates
  - Works with gamma to balance short-term vs long-term learning
- `clip_range` (0.1 to 0.3): PPO clipping parameter to prevent destructive policy updates
  - Higher = allows bigger policy changes (more aggressive updates)
  - Lower = smaller, safer updates

**Data Collection:**
- `n_steps` (128 to 1024): Experience buffer size per environment before policy update
  - In Big Two terms: ~3-30 complete games worth of moves (games are ~20-40 moves)
  - Higher = more data per update, more stable but slower learning
- `n_envs` (2 to 16): How many parallel Big Two tables running simultaneously
  - Like having multiple different card tables generating training data at once
  - Derived from CPU count with different ratios per config
- `batch_size` (16 to 128): How many individual moves to review together when updating strategy
  - Should be ≤ `n_steps * n_envs` (total collected experience)
  - Larger batches = more stable gradients but more memory usage
- `n_epochs` (3 to 15): How many times to review the same batch of moves
  - Like replaying the same hands multiple times to learn from them
  - Higher = extract more learning from same data but risk overfitting

**Game-Specific:**
- `games_per_episode` (2 to 10): How many complete Big Two games before the AI gets feedback
  - Why multiple games? Card dealing is random, so one game isn't enough signal
  - Agent only gets reward at episode end (after all games)

### Parameter Relationships & Dependencies

**Derived Parameters:**
- `n_envs`: All configs derive from CPU count with different formulas
  - Default/Conservative: `cpu_count // 2` 
  - Aggressive: `cpu_count // 3`
  - FastExperimental: Uses more CPUs for speed

**Critical Relationships:**
- `batch_size` ≤ `n_steps * n_envs`: Batch can't be larger than total collected data
- `learning_rate` × `clip_range`: Related aggression levels (both high in aggressive config)
- `gamma` × `gae_lambda`: Both affect temporal credit assignment at different stages

**Training Volume:**
- `total_timesteps` (25,000): Total individual card plays for entire training
  - At ~30 moves/game and 5 games/episode = ~150 moves/episode
  - So 25k timesteps ≈ 167 episodes ≈ 835 total games

### Configuration Profiles

**DefaultConfig**: Balanced training for general use
- Moderate learning rate (3e-4), stable gamma (0.99)
- 512 steps, 64 batch size, 10 epochs, 5 games/episode

**AggressiveConfig**: Faster, less stable training
- High learning rate (1e-3), lower gamma (0.95), high clip_range (0.3)
- 256 steps, 32 batch size, 5 epochs, 3 games/episode

**ConservativeConfig**: Stable, slower training
- Low learning rate (1e-4), high gamma (0.995), low clip_range (0.1)
- 1024 steps, 128 batch size, 15 epochs, 10 games/episode

**FastExperimentalConfig**: Quick testing and iteration
- High learning rate (5e-4), low gamma (0.9)
- 128 steps, 16 batch size, 3 epochs, 2 games/episode  

## Multiprocessing Support

### Parallel Tournament Execution
The library now supports multiprocessing for significant tournament speedup:

**Core Features**:
- **Game-Level Parallelization**: Split games across multiple CPU cores  
- **Automatic Process Management**: Auto-detects optimal process count
- **Agent Serialization**: Handles PPO model loading across processes
- **Result Aggregation**: Properly merges statistics from parallel workers

**Performance Gains**:
- **2-3x speedup** with 4 processes on typical systems
- **Best for large tournaments**: 200+ games see significant benefits
- **Fallback handling**: Automatically uses sequential for small runs (<10 games)

**Usage Examples**:
```python
# Automatic multiprocessing (recommended)
tournament = Tournament(agents, n_processes=None)  # Auto-detects CPUs
results = tournament.run(num_games=1000)

# Explicit process control  
tournament = Tournament(agents, n_processes=4)
results = tournament.run(num_games=500)

# Sequential fallback
tournament = Tournament(agents, n_processes=1)  # Force single process
results = tournament.run(num_games=100)

# Evaluator with multiprocessing
evaluator = Evaluator(num_games=500, n_processes=4)
results = evaluator.evaluate_model("./models/my_model")
```

**Technical Implementation**:
- Uses `multiprocessing.Pool` with worker batches
- Agent serialization supports RandomAgent, GreedyAgent, PPOAgent  
- Different random seeds per process ensure reproducible variety
- Aggregates wins, card statistics, and game history across processes

**Limitations**:
- PPO agents must be created with `model_path` (not in-memory models)
- Agent stats (wins/games_played) are approximate after parallel execution
- Process overhead makes single games slower than sequential

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

### Quick Start Training Pattern
```python
# 1. Choose components
from bigtwo_rl.training import Trainer
from bigtwo_rl.training.rewards import DefaultReward  # or any of the 8 reward functions
from bigtwo_rl.training.hyperparams import DefaultConfig  # or any of the 4 configs
from bigtwo_rl import standard_observation  # or minimal/memory_enhanced/strategic

# 2. Configure trainer  
trainer = Trainer(
    reward_function=DefaultReward(),
    hyperparams=DefaultConfig(), 
    observation_config=standard_observation()
)

# 3. Train model
model, model_dir = trainer.train(total_timesteps=25000)

# 4. Evaluate model (automatically loads the best model)
from bigtwo_rl.evaluation import Evaluator
evaluator = Evaluator(num_games=100)
results = evaluator.evaluate_model(f"{model_dir}/best_model")
print(f"Win rate: {results['win_rate']:.2%}")
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
    def game_reward(self, winner_player, player_idx, cards_left, all_cards_left=None):
        # Your reward logic here
        return reward_value
        
    def episode_bonus(self, games_won, total_games, avg_cards_left):
        # Episode-level bonus
        return bonus_value
        
    def move_bonus(self, move_cards):
        # Optional: bonus for specific moves  
        return move_bonus_value
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