# Big Two RL Agent Library

A comprehensive reinforcement learning library for training AI agents to play Big Two (Chinese card game) using PPO (Proximal Policy Optimization). This library provides everything needed for Big Two AI research: training, evaluation, tournaments, and detailed performance analysis.

## 🚀 Key Features

- **🎯 Complete Training Pipeline**: End-to-end training from scratch to tournament-ready agents
- **⚡ High-Performance Core**: Vectorized numpy implementation (5-20x speedup, 8,000+ steps/sec)
- **🔧 Flexible Configuration**: Multiple hyperparameter presets and custom reward functions
- **🏆 Tournament System**: Round-robin tournaments with detailed statistics and multiprocessing
- **📊 Comprehensive Evaluation**: Win rates, performance metrics, and learning curve analysis
- **🧠 Multiple Agent Types**: Random, Greedy, and PPO agents with consistent interfaces
- **💾 Memory Efficient**: Optimized data structures (50-75% memory reduction)
- **🎮 Interactive Play**: Play against trained agents for testing and demo purposes

## Installation

### Development Install
```bash
# Clone the repository
git clone <repository-url>
cd bigtwo-rl

# Install dependencies
uv sync

# Install library in development mode
pip install -e .
```

### Production Install
```bash
pip install bigtwo-rl
```

## 🚦 Quick Start

### 1. Training Your First Agent
```python
from bigtwo_rl.training import Trainer

# Basic training (25k timesteps ≈ 3 minutes)
trainer = Trainer()
model, model_dir = trainer.train(total_timesteps=25000)
print(f"✅ Model saved to: {model_dir}")

# Advanced training with custom settings
trainer = Trainer(
    hyperparams="aggressive",           # Faster learning
    reward_function="progressive",      # Rewards card reduction
    save_every=5000,                   # Checkpoint frequency
    verbose=1                          # Show training progress
)
model, model_dir = trainer.train(total_timesteps=50000, description="Aggressive Progressive Training")
```

### 2. Evaluating Your Agent
```python
from bigtwo_rl.evaluation import Evaluator

# Standard 4-player evaluation
evaluator = Evaluator(num_games=100, verbose=True)
results = evaluator.evaluate_model("./models/my_model/best_model")

print(f"🏆 Win Rate: {results['win_rates'][0]:.1%}")
print(f"🎯 Avg Cards Left: {results['avg_cards_left'][0]:.1f}")
print(f"📊 Games Won: {results['wins'][0]}/{results['games_played']}")
```

### 3. Tournament Competition
```python
from bigtwo_rl.agents import RandomAgent, GreedyAgent, PPOAgent
from bigtwo_rl.evaluation import Tournament

# Set up 4-player tournament
agents = [
    PPOAgent("./models/my_model/best_model", "MyAgent"),
    GreedyAgent("Greedy"),
    RandomAgent("Random-1"),
    RandomAgent("Random-2"),
]

# Run tournament with multiprocessing (auto-detects CPUs)
tournament = Tournament(agents, n_processes=None)
results = tournament.run_round_robin(num_games=200)

print("🏆 Tournament Results:")
for agent, wins in zip(results["agents"], results["total_wins"]):
    win_rate = wins / results["total_games"] * 100
    print(f"  {agent}: {wins} wins ({win_rate:.1f}%)")
```

## Library Structure

```
bigtwo_rl/                           # Main library package
├── __init__.py                      # Main exports
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
│   └── rewards.py                  # Reward functions
├── evaluation/                      # Evaluation and competition
│   ├── evaluator.py                # Model assessment (4-player series)
│   └── tournament.py               # 4-player series and tournaments
└── utils/                           # Utilities and helpers
```

## Usage Examples

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

### Custom Agent
```python
from bigtwo_rl.agents import BaseAgent

class MyAgent(BaseAgent):
    def get_action(self, observation, action_mask=None):
        # Your agent logic here
        return action_index
    
    def reset(self):
        # Reset agent state if needed
        pass
```

### Advanced Training Configuration
```python
from bigtwo_rl.training import Trainer

trainer = Trainer(
    hyperparams="default",      # or "aggressive", "conservative", "fast_experimental"
    reward_function="score_margin",  # also: "progressive", "ranking", ...
    # Self-play improvements (optional):
    controlled_player=0,  # seat controlled by PPO
    opponent_mixture={"snapshots": 0.6, "greedy": 0.3, "random": 0.1},
    snapshot_dir="./models/my_run",  # discover/save snapshots
    snapshot_every_steps=10000,       # periodically save snapshots
    eval_freq=5000,
    verbose=1
)

model, model_dir = trainer.train(
    total_timesteps=50000,
    description="Experimental training run"
)
```

## Built-in Components

### Reward Functions
- **default**: Win +5, loss penalty scaled by remaining cards
- **sparse**: Simple win (+1) vs loss (-1) 
- **aggressive_penalty**: Higher penalties for losing with many cards
- **progressive**: Rewards progress (fewer cards = better reward)
- **ranking**: Rewards based on final ranking among all players
 - **score_margin**: Continuous reward blending win/loss with normalized card-margin vs opponents

### Hyperparameter Configurations
- **default**: Balanced settings for general training
- **aggressive**: Higher learning rates and penalties
- **conservative**: More stable, slower learning
- **fast_experimental**: Quick experimentation settings

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

---

## 🎯 Training Guide

### Basic Training Workflow

The training process consists of three main steps: configuration, training, and evaluation.

```python
from bigtwo_rl.training import Trainer
from bigtwo_rl.evaluation import Evaluator

# Step 1: Configure training
trainer = Trainer(
    hyperparams="default",              # Choose preset: default, aggressive, conservative, fast_experimental
    reward_function="default",          # Choose reward: default, progressive, ranking, sparse
    save_every=10000,                  # Save checkpoint every N steps
    verbose=1                          # Show progress: 0=silent, 1=progress, 2=detailed
)

# Step 2: Train the model
model, model_dir = trainer.train(
    total_timesteps=50000,             # Total training steps (50k ≈ 6 minutes)
    description="My training run"       # Optional description for model directory
)

# Step 3: Evaluate performance
evaluator = Evaluator(num_games=100)
results = evaluator.evaluate_model(f"{model_dir}/best_model")
print(f"Win rate: {results['win_rates'][0]:.1%}")
```

### Hyperparameter Configurations

Choose from four optimized presets, each designed for different training goals:

| Preset | Learning Rate | Training Speed | Stability | Best For |
|--------|---------------|----------------|-----------|----------|
| **`default`** | 3e-4 | Moderate (3 min/25k) | ⭐⭐⭐⭐ | General purpose, first training runs |
| **`aggressive`** | 1e-3 | Fast (2 min/25k) | ⭐⭐⭐ | Quick experiments, when you need fast results |
| **`conservative`** | 1e-4 | Slow (6 min/25k) | ⭐⭐⭐⭐⭐ | Stable training, fine-tuning, research |
| **`fast_experimental`** | 5e-4 | Very Fast (1 min/25k) | ⭐⭐ | Rapid prototyping, hyperparameter search |

#### Detailed Configuration Parameters

```python
# Example: Custom hyperparameter override
from bigtwo_rl.training.hyperparams import get_config

config = get_config("default")
config.update({
    "learning_rate": 5e-4,      # PPO learning rate
    "n_steps": 256,             # Steps per update (256 = ~8-12 games)
    "batch_size": 32,           # Batch size for gradient updates
    "n_epochs": 8,              # Training epochs per update
    "gamma": 0.99,              # Reward discount factor
    "gae_lambda": 0.95,         # GAE lambda for advantage estimation
    "clip_range": 0.2,          # PPO clip range
    "games_per_episode": 5,     # Games per RL episode
    "n_envs": 8,                # Parallel environments
})

trainer = Trainer(hyperparams=config)
```

### Reward Functions

Choose the reward structure that matches your training goals:

#### Built-in Reward Functions

```python
# Available reward functions
reward_options = [
    "default",           # Balanced: Win +1, loss penalty by cards left
    "sparse",            # Simple: Win +1, loss -1
    "progressive",       # Rewards card reduction progress
    "ranking",           # Rewards based on final ranking (1st, 2nd, 3rd, 4th)
    "aggressive_penalty" # High penalties for losing with many cards
]

trainer = Trainer(reward_function="progressive")  # Use by name
```

#### Custom Reward Functions

Create sophisticated reward structures for specific training objectives:

```python
from bigtwo_rl.training.rewards import BaseReward

class StrategicReward(BaseReward):
    """Reward function that encourages aggressive early play and card conservation."""
    
    def game_reward(self, winner_player, player_idx, cards_left, all_cards_left=None):
        if player_idx == winner_player:
            return 2.0  # Strong win reward
        
        # Penalty increases exponentially with cards left
        penalty = -(cards_left / 13) ** 1.5
        
        # Bonus for beating other players (ranking matters)
        better_than = sum(1 for other_cards in all_cards_left if other_cards > cards_left)
        ranking_bonus = better_than * 0.2
        
        return penalty + ranking_bonus
    
    def episode_bonus(self, games_won, total_games, avg_cards_left):
        # Consistency bonus for winning multiple games
        win_rate = games_won / total_games
        consistency_bonus = win_rate ** 2 * 0.5
        
        # Efficiency bonus for low average cards left
        efficiency_bonus = max(0, (10 - avg_cards_left) * 0.1)
        
        return consistency_bonus + efficiency_bonus

# Use custom reward function
trainer = Trainer(reward_function=StrategicReward())
```

### Advanced Training Features

#### Self-Play Training

Train against evolving opponents for more robust learning:

```python
trainer = Trainer(
    hyperparams="default",
    reward_function="progressive",
    # Self-play configuration
    controlled_player=0,                    # Which seat the PPO agent controls
    opponent_mixture={                      # Mix of opponent types
        "snapshots": 0.6,                  # 60% past versions of self
        "greedy": 0.3,                     # 30% greedy baseline
        "random": 0.1                      # 10% random
    },
    snapshot_dir="./models/my_self_play",  # Where to save/load snapshots
    snapshot_every_steps=10000,            # Save model every 10k steps
    eval_freq=5000,                        # Evaluate every 5k steps
    verbose=1
)

model, model_dir = trainer.train(
    total_timesteps=100000,  # Longer training for self-play
    description="Self-play training run"
)
```

#### Training Monitoring

Track training progress with built-in logging and TensorBoard:

```bash
# Start TensorBoard (run in separate terminal)
uv run tensorboard --logdir=./logs

# Then visit: http://localhost:6006
```

```python
# Training with detailed logging
trainer = Trainer(
    hyperparams="default",
    reward_function="progressive",
    save_every=5000,        # Frequent checkpoints
    verbose=2               # Detailed logging
)

# Tensorboard will show:
# - Episode rewards over time
# - Win rates vs different opponents  
# - Training loss and policy entropy
# - Average episode length (game efficiency)
```

---

## 📊 Evaluation & Analysis

### Comprehensive Model Evaluation

The `Evaluator` class provides detailed analysis of your trained models:

```python
from bigtwo_rl.evaluation import Evaluator

# Basic evaluation
evaluator = Evaluator(
    num_games=500,          # More games = more reliable statistics
    verbose=True,           # Show progress bar
    opponent_agents=None    # Use default mix of opponents
)

results = evaluator.evaluate_model("./models/my_model/best_model")

# Detailed results breakdown
print("🎯 Performance Summary:")
print(f"  Win Rate: {results['win_rates'][0]:.1%}")
print(f"  Wins: {results['wins'][0]}/{results['games_played']}")
print(f"  Avg Cards Left: {results['avg_cards_left'][0]:.2f}")
print(f"  Opponent Win Rates: {[f'{wr:.1%}' for wr in results['win_rates'][1:]]}")

# Game-by-game analysis
print(f"\n📈 Performance Distribution:")
cards_left_history = results['cards_left_by_game'][0]  # Cards left in each game
wins_history = [1 if cards == 0 else 0 for cards in cards_left_history]

# Calculate rolling win rate
window_size = 50
rolling_wins = []
for i in range(window_size, len(wins_history)):
    rolling_win_rate = sum(wins_history[i-window_size:i]) / window_size
    rolling_wins.append(rolling_win_rate)

print(f"  Early games (1-50): {sum(wins_history[:50])/50:.1%} win rate")
print(f"  Late games ({len(wins_history)-50}+): {sum(wins_history[-50:])/50:.1%} win rate")
```

### Custom Evaluation Opponents

Test your agent against specific opponent configurations:

```python
from bigtwo_rl.agents import RandomAgent, GreedyAgent, PPOAgent

# Test against specific opponents
custom_opponents = [
    GreedyAgent("Greedy-1"),
    GreedyAgent("Greedy-2"),  
    RandomAgent("Random-1")
]

evaluator = Evaluator(
    num_games=200,
    opponent_agents=custom_opponents,  # Use custom opponent mix
    verbose=True
)

results = evaluator.evaluate_model("./models/my_model/best_model")
print(f"vs Greedy+Random: {results['win_rates'][0]:.1%} win rate")

# Compare against another trained model
strong_opponents = [
    PPOAgent("./models/baseline_model/best_model", "Baseline"),
    GreedyAgent("Greedy"),
    RandomAgent("Random")
]

evaluator = Evaluator(num_games=300, opponent_agents=strong_opponents)
results = evaluator.evaluate_model("./models/new_model/best_model")
print(f"vs Strong Baseline: {results['win_rates'][0]:.1%} win rate")
```

---

## 🏆 Tournament System

### Round-Robin Tournaments

The tournament system runs comprehensive competitions between multiple agents:

```python
from bigtwo_rl.agents import RandomAgent, GreedyAgent, PPOAgent
from bigtwo_rl.evaluation import Tournament

# Set up tournament participants
agents = [
    PPOAgent("./models/agent_v1/best_model", "Agent-v1"),
    PPOAgent("./models/agent_v2/best_model", "Agent-v2"),
    GreedyAgent("Greedy"),
    RandomAgent("Random")
]

# Run comprehensive tournament
tournament = Tournament(
    agents=agents,
    n_processes=None,       # Auto-detect CPUs for parallel execution
    verbose=True           # Show detailed progress
)

# Execute round-robin (every possible 4-agent combination)
results = tournament.run_round_robin(num_games=500)
```

### Tournament Results Analysis

```python
# Overall tournament summary
print("🏆 Tournament Results:")
print(results["tournament_summary"])

# Detailed breakdowns
print("\n📊 Detailed Statistics:")
total_games = results["total_games"]
for i, agent_name in enumerate(results["agents"]):
    wins = results["total_wins"][i]
    win_rate = wins / total_games
    avg_cards = results["avg_cards_left"][i]
    
    print(f"  {agent_name:15} | {wins:4d} wins ({win_rate:.1%}) | {avg_cards:.1f} avg cards left")

# Head-to-head performance matrix
print(f"\n🥊 Head-to-Head Breakdown:")
matchup_results = results["matchup_results"]
for matchup, stats in matchup_results.items():
    print(f"  {matchup}: {stats}")
```

### High-Performance Tournament Features

#### Multiprocessing Support

Leverage multiple CPU cores for faster tournaments:

```python
import os

# Automatic CPU detection (recommended)
tournament = Tournament(agents, n_processes=None)  # Uses all available CPUs

# Manual process control
tournament = Tournament(agents, n_processes=4)     # Use exactly 4 processes

# Single-threaded (useful for debugging)
tournament = Tournament(agents, n_processes=1)

# Performance comparison
print(f"Available CPUs: {os.cpu_count()}")
print("Tournament scales best with 200+ games")

# Large tournament example (2-3x speedup with multiprocessing)
results = tournament.run_round_robin(
    num_games=1000,        # Large tournaments benefit most from parallelization
    verbose=True           # Shows progress across processes
)
```

#### Custom Tournament Formats

Create specialized tournament formats:

```python
# Single-elimination style (custom logic)
def elimination_tournament(agents, games_per_round=100):
    """Custom tournament format example."""
    current_agents = agents.copy()
    round_num = 1
    
    while len(current_agents) > 1:
        print(f"\n🔥 Round {round_num} ({len(current_agents)} agents)")
        
        # Pair agents randomly and compete
        tournament = Tournament(current_agents[:4])  # Take first 4 for demo
        results = tournament.run_round_robin(num_games=games_per_round)
        
        # Eliminate bottom performers (keep top 50%)
        agent_performance = list(zip(results["agents"], results["total_wins"]))
        agent_performance.sort(key=lambda x: x[1], reverse=True)
        
        survivors = agent_performance[:len(agent_performance)//2]
        current_agents = [PPOAgent(f"./models/{name}/best_model", name) 
                         for name, _ in survivors if "Agent" in name]
        
        round_num += 1
        
    return current_agents[0] if current_agents else None

# Run custom tournament
winner = elimination_tournament(agents, games_per_round=200)
print(f"🏆 Tournament Champion: {winner.name if winner else 'None'}")
```

### Performance Benchmarking

Use tournaments to benchmark training improvements:

```python
# Benchmark different training approaches
models_to_test = [
    ("./models/default_training/best_model", "Default"),
    ("./models/aggressive_training/best_model", "Aggressive"), 
    ("./models/self_play_training/best_model", "Self-Play"),
    ("./models/custom_reward_training/best_model", "Custom-Reward")
]

benchmark_agents = []
for model_path, name in models_to_test:
    benchmark_agents.append(PPOAgent(model_path, name))

# Add consistent baselines
benchmark_agents.extend([
    GreedyAgent("Greedy-Baseline"),
    RandomAgent("Random-Baseline")
])

# Run comprehensive benchmark
tournament = Tournament(benchmark_agents, n_processes=None)
benchmark_results = tournament.run_round_robin(num_games=400)

# Analyze which training approach works best
print("🧪 Training Method Benchmark:")
for name, wins in zip(benchmark_results["agents"], benchmark_results["total_wins"]):
    if "Baseline" not in name:
        win_rate = wins / benchmark_results["total_games"]
        print(f"  {name:15}: {win_rate:.1%} win rate")
```

---

## ⚡ Performance & Optimization

### High-Performance Features

This library is optimized for fast RL training and evaluation:

```python
# Performance metrics (typical laptop)
performance_stats = {
    "Environment reset": "0.044ms (22,775 resets/sec)",
    "Game steps": "0.118ms (8,475 steps/sec)",
    "Legal move generation": "0.038ms (26,040 calls/sec)", 
    "Training throughput": "8,000+ steps/sec",
    "Memory usage": "50-75% reduction vs list-based implementation"
}

# Benchmarking your setup
from bigtwo_rl.core import BigTwoRLWrapper
import time

env = BigTwoRLWrapper()
num_tests = 10000

# Test reset performance
start = time.time()
for _ in range(num_tests):
    env.reset()
reset_time = (time.time() - start) / num_tests
print(f"Reset performance: {reset_time*1000:.3f}ms per reset")

# Test step performance  
env.reset()
start = time.time()
for _ in range(num_tests):
    action = env.action_space.sample()
    env.step(action)
    if env.terminated or env.truncated:
        env.reset()
step_time = (time.time() - start) / num_tests
print(f"Step performance: {step_time*1000:.3f}ms per step")
```

### Memory Optimization

The library uses optimized data structures for memory efficiency:

```python
# Memory-efficient data structures
import numpy as np

# Boolean array representation (50% memory savings)
hand_representation = np.array([True, False, True, ...], dtype=bool)  # 52 bools per hand
memory_per_hand = hand_representation.nbytes  # 52 bytes vs 104+ bytes for int arrays

# Vectorized operations (no intermediate allocations)
legal_moves = np.where(action_mask)[0]  # Direct numpy indexing
hand_types = identify_hand_type_vectorized(hand_array)  # Batch processing

print(f"Memory per game state: ~{52*4 + 52 + 4 + 1} bytes")  # Hands + last play + sizes + flag
print(f"Memory reduction: 50-75% vs traditional implementations")
```

### Training Performance Tips

Optimize your training for maximum speed:

```python
# High-performance training configuration
trainer = Trainer(
    hyperparams={
        "n_envs": min(16, os.cpu_count()),    # Use all CPUs
        "n_steps": 256,                       # Moderate batch size
        "batch_size": 64,                     # Efficient GPU utilization
        "games_per_episode": 3,               # Faster episode completion
    },
    save_every=20000,                        # Less frequent I/O
    verbose=1                               # Minimal logging overhead
)

# Monitor training efficiency
import time
start_time = time.time()
model, model_dir = trainer.train(total_timesteps=25000)
training_duration = time.time() - start_time

steps_per_second = 25000 / training_duration
print(f"Training speed: {steps_per_second:.0f} steps/sec")
print(f"Expected: 8,000+ steps/sec on modern hardware")
```

## Development Commands

```bash
# Training examples
uv run python examples/train_agent.py               # Train agent with simple API
uv run python examples/evaluate_agent.py MODEL     # Evaluate trained model (4-player series)
uv run python examples/tournament_example.py       # Run 4-player tournament
uv run python examples/custom_reward_example.py    # Custom reward training

# Testing
uv run python tests/test_wrapper.py        # Test environment wrapper
uv run python tests/test_cards.py          # Test card utilities (including numpy arrays)
uv run python tests/test_rewards.py        # Test reward functions
uv run python tests/test_training.py       # Test training setup
uv run python tests/test_evaluation.py     # Test evaluation and tournament systems
uv run python tests/test_numpy_performance.py  # Test performance optimizations

# Interactive play
uv run python examples/play_vs_agent.py MODEL      # Play against trained agent

# Monitoring
tensorboard --logdir=./logs                        # View training metrics
```

## Game Environment Details

### Observation Space (109 features)
- Hand binary encoding (52 features)
- Last play binary encoding (52 features) 
- Hand sizes for all players (4 features)
- Last play exists flag (1 feature)

### Action Space
Dynamic action space based on legal moves:
- Single cards, pairs, trips
- 5-card hands (straights, flushes, full houses, etc.)
- Pass action (when not starting a trick)

### Big Two Rules
- **Suits**: Diamonds < Clubs < Hearts < Spades
- **Ranks**: 3 < 4 < 5 < 6 < 7 < 8 < 9 < 10 < J < Q < K < A < 2
- **Goal**: First player to play all cards wins

## Performance & Training Results

### High-Performance Optimizations
- **Vectorized Operations**: Fully numpy-based card operations and game logic
- **Memory Efficiency**: 50-75% reduction in memory usage with boolean arrays
- **Training Speed**: 8,000+ game steps per second for fast RL training
- **Legal Move Generation**: 26,000+ calls per second with vectorized algorithms

### Training Results
- 100% win rate vs random and greedy baselines after 50k timesteps
- Episode length reduction from 69→43 steps (learned efficiency)
- Successful convergence with full Big Two complexity
- Stable training across different reward functions
- **Fast Training**: Optimized for high-throughput reinforcement learning

## API Reference

### Main Classes

#### `Trainer`
```python
Trainer(hyperparams="default", reward_function="default", save_every=10000, verbose=1)
```
High-level training interface with configurable parameters.

#### `Evaluator` 
```python
Evaluator(num_games=50)
```
Evaluate trained models in 4-player series against three baseline opponents.

#### `Tournament`
```python
Tournament(agents, verbose=True)
```
Run 4-player competitions between multiple agents (round-robin across 4-agent tables).

#### `BigTwoRLWrapper`
```python
BigTwoRLWrapper(
    num_players=4,
    games_per_episode=5,
    reward_function=None,
    controlled_player=0,
    opponent_provider=None,
)
```
Gymnasium-compatible environment for RL training.

### Self-Play and Opponent Pool (Clean API)

Configure self-play via `Trainer` arguments only. Opponents are auto-stepped inside the env until it’s the learner’s turn; PPO still sees a standard `gym.Env`.

- `controlled_player` (int): seat index controlled by PPO (default 0).
- `opponent_mixture` (dict): sampling weights for {"snapshots", "greedy", "random"}.
- `snapshot_dir` (str): directory to discover and save snapshots.
- `snapshot_every_steps` (int): frequency to snapshot current model.

Quick smoke:
```bash
uv run python examples/smoke_train.py
uv run python - <<'PY'
from bigtwo_rl.evaluation.evaluator import evaluate_agent
print(evaluate_agent('./models/smoke_test/final_model', num_games=100))
PY
```

---

## 🛠️ Troubleshooting & Advanced Usage

### Common Issues and Solutions

#### Training Issues

**Problem**: Training is slow or not converging
```python
# Solution 1: Adjust hyperparameters for faster convergence
trainer = Trainer(
    hyperparams="aggressive",        # Faster learning rate
    reward_function="progressive",   # More informative rewards
    verbose=2                       # Enable detailed logging
)

# Solution 2: Increase parallel environments
config = get_config("default")
config["n_envs"] = min(16, os.cpu_count())  # Use more CPUs
trainer = Trainer(hyperparams=config)

# Solution 3: Monitor with TensorBoard
# uv run tensorboard --logdir=./logs
# Check for: reward trends, episode lengths, policy entropy
```

**Problem**: Agent learns to play very conservatively (high win rate but boring play)
```python
# Solution: Use reward functions that encourage aggressive play
trainer = Trainer(
    reward_function="ranking",       # Rewards beating other players
    hyperparams="aggressive"        # Higher learning rate
)

# Or create custom reward that penalizes holding cards
class AggressiveReward(BaseReward):
    def game_reward(self, winner_player, player_idx, cards_left, all_cards_left=None):
        if player_idx == winner_player:
            return 2.0 + (13 - cards_left) * 0.1  # Bonus for winning with fewer moves
        return -(cards_left ** 1.5) * 0.1  # Strong penalty for many remaining cards
```

**Problem**: Training terminates with errors
```python
# Common fix: Ensure proper environment cleanup
try:
    trainer = Trainer(hyperparams="default", verbose=1)
    model, model_dir = trainer.train(total_timesteps=25000)
except Exception as e:
    print(f"Training error: {e}")
    # Check: sufficient disk space, proper uv environment, Python version 3.8+
```

#### Evaluation Issues

**Problem**: Evaluation results seem inconsistent
```python
# Solution: Use more games and consistent opponents
evaluator = Evaluator(
    num_games=500,              # More games = more reliable stats
    opponent_agents=[           # Consistent opponent pool
        GreedyAgent("Greedy"),
        RandomAgent("Random-1"),
        RandomAgent("Random-2")
    ],
    verbose=True
)

# Run multiple evaluations to check consistency
results_1 = evaluator.evaluate_model("./models/my_model/best_model")
results_2 = evaluator.evaluate_model("./models/my_model/best_model")
print(f"Run 1: {results_1['win_rates'][0]:.1%}, Run 2: {results_2['win_rates'][0]:.1%}")
```

**Problem**: Tournament results don't match expectations
```python
# Solution: Check for proper agent loading and multiprocessing issues
tournament = Tournament(agents, n_processes=1)  # Disable multiprocessing for debugging
results = tournament.run_round_robin(num_games=100)

# Verify agent implementations
for agent in agents:
    if hasattr(agent, 'model_path'):
        print(f"{agent.name}: {agent.model_path}")
    else:
        print(f"{agent.name}: Built-in agent")
```

#### Performance Issues

**Problem**: Training is slower than expected
```python
# Benchmark your system
from bigtwo_rl.core import BigTwoRLWrapper
import time

env = BigTwoRLWrapper()
start = time.time()
for _ in range(1000):
    env.reset()
    for _ in range(50):  # Average game length
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

elapsed = time.time() - start
steps_per_sec = (1000 * 50) / elapsed
print(f"Your system: {steps_per_sec:.0f} steps/sec")
print(f"Expected: 8,000+ steps/sec")

# If slow, try:
# 1. Reduce n_envs in hyperparameters
# 2. Check Python version (3.9+ recommended)  
# 3. Ensure numpy is properly installed
```

### Advanced Customization

#### Custom Agent Development

```python
from bigtwo_rl.agents import BaseAgent
import numpy as np

class RuleBasedAgent(BaseAgent):
    """Example: Rule-based agent with heuristics."""
    
    def __init__(self, name="RuleBased"):
        super().__init__(name)
        self.play_style = "aggressive"  # or "conservative"
    
    def get_action(self, observation, action_mask=None):
        """Implement custom decision logic."""
        if action_mask is None:
            return 0  # Pass or default action
            
        legal_actions = np.where(action_mask)[0]
        if len(legal_actions) == 0:
            return 0
            
        # Custom heuristics
        if self.play_style == "aggressive":
            # Always play highest legal card
            return legal_actions[-1]
        else:
            # Play lowest legal card (greedy strategy)
            return legal_actions[0]
    
    def reset(self):
        """Reset any internal state."""
        pass

# Use in tournaments
agents = [
    RuleBasedAgent("Rule-Aggressive"),
    RuleBasedAgent("Rule-Conservative"),
    GreedyAgent("Greedy"),
    RandomAgent("Random")
]

tournament = Tournament(agents)
results = tournament.run_round_robin(num_games=200)
```

#### Custom Environment Modifications

```python
from bigtwo_rl.core import BigTwoRLWrapper

class CustomBigTwoEnv(BigTwoRLWrapper):
    """Example: Modified environment with custom features."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.move_history = []  # Track all moves
        self.game_statistics = {"total_games": 0, "avg_game_length": 0}
    
    def step(self, action):
        """Override step to add custom logging/features."""
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Log move history
        self.move_history.append({
            "action": action,
            "reward": reward,
            "terminated": terminated
        })
        
        # Update statistics
        if terminated or truncated:
            self.game_statistics["total_games"] += 1
            game_length = len(self.move_history)
            
            # Update running average
            n = self.game_statistics["total_games"]
            current_avg = self.game_statistics["avg_game_length"]
            self.game_statistics["avg_game_length"] = ((n-1) * current_avg + game_length) / n
            
            self.move_history = []  # Reset for next game
        
        return obs, reward, terminated, truncated, info
    
    def get_statistics(self):
        """Return custom statistics."""
        return self.game_statistics.copy()

# Use custom environment in training
trainer = Trainer(env_class=CustomBigTwoEnv)
model, model_dir = trainer.train(total_timesteps=25000)
```

#### Multi-Stage Training Pipeline

```python
def multi_stage_training(total_timesteps=100000):
    """Example: Progressive training with different stages."""
    
    # Stage 1: Basic learning against random opponents
    print("🎯 Stage 1: Basic Learning (vs Random)")
    stage1_trainer = Trainer(
        hyperparams="fast_experimental",
        reward_function="sparse",
        verbose=1
    )
    model1, dir1 = stage1_trainer.train(
        total_timesteps=total_timesteps // 4,
        description="Stage1_Basic"
    )
    
    # Evaluate stage 1
    evaluator = Evaluator(num_games=100)
    stage1_results = evaluator.evaluate_model(f"{dir1}/best_model")
    print(f"Stage 1 results: {stage1_results['win_rates'][0]:.1%} win rate")
    
    # Stage 2: Intermediate learning with better rewards
    print("🎯 Stage 2: Intermediate Learning (Progressive Rewards)")
    stage2_trainer = Trainer(
        hyperparams="default",
        reward_function="progressive",
        verbose=1
    )
    # Load stage 1 model as starting point
    # (Note: This requires additional implementation for model loading)
    model2, dir2 = stage2_trainer.train(
        total_timesteps=total_timesteps // 2,
        description="Stage2_Progressive"
    )
    
    # Stage 3: Advanced learning with self-play
    print("🎯 Stage 3: Advanced Learning (Self-Play)")
    stage3_trainer = Trainer(
        hyperparams="conservative",
        reward_function="ranking",
        controlled_player=0,
        opponent_mixture={"snapshots": 0.5, "greedy": 0.3, "random": 0.2},
        snapshot_dir=dir2,
        snapshot_every_steps=5000,
        verbose=1
    )
    model3, dir3 = stage3_trainer.train(
        total_timesteps=total_timesteps // 4,
        description="Stage3_SelfPlay"
    )
    
    # Final evaluation
    final_results = evaluator.evaluate_model(f"{dir3}/best_model")
    print(f"🏆 Final results: {final_results['win_rates'][0]:.1%} win rate")
    
    return model3, dir3

# Run multi-stage training
final_model, final_dir = multi_stage_training(total_timesteps=50000)
```

### Interactive Development

#### Jupyter Notebook Integration

```python
# Cell 1: Setup and training
from bigtwo_rl.training import Trainer
from bigtwo_rl.evaluation import Evaluator
import matplotlib.pyplot as plt

trainer = Trainer(hyperparams="fast_experimental", verbose=1)
model, model_dir = trainer.train(total_timesteps=10000)

# Cell 2: Quick evaluation
evaluator = Evaluator(num_games=50)
results = evaluator.evaluate_model(f"{model_dir}/best_model")
print(f"Win rate: {results['win_rates'][0]:.1%}")

# Cell 3: Visualization (if you have matplotlib)
cards_left_data = results['cards_left_by_game'][0]
plt.hist(cards_left_data, bins=14, alpha=0.7, edgecolor='black')
plt.xlabel('Cards Left at End of Game')
plt.ylabel('Frequency')
plt.title('Agent Performance Distribution')
plt.show()
```

#### Live Training Monitoring

```python
import threading
import time
from bigtwo_rl.training import Trainer

def monitor_training(model_dir, check_interval=30):
    """Monitor training progress in real-time."""
    while True:
        try:
            # Check if model exists and evaluate
            model_path = f"{model_dir}/best_model"
            if os.path.exists(f"{model_path}.zip"):
                evaluator = Evaluator(num_games=20, verbose=False)
                results = evaluator.evaluate_model(model_path)
                win_rate = results['win_rates'][0]
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] Current win rate: {win_rate:.1%}")
            
            time.sleep(check_interval)
        except Exception as e:
            print(f"Monitoring error: {e}")
            break

# Start monitoring in background thread
trainer = Trainer(hyperparams="default", save_every=5000, verbose=1)

# Start monitoring thread
monitor_thread = threading.Thread(
    target=monitor_training, 
    args=("./models", 30),
    daemon=True
)
monitor_thread.start()

# Run training
model, model_dir = trainer.train(total_timesteps=50000)
```

### Debugging and Development Tips

#### Enable Detailed Logging

```python
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# This will show internal library operations
trainer = Trainer(hyperparams="default", verbose=2)
```

#### Reproducible Results

```python
import random
import numpy as np
import torch

def set_random_seeds(seed=42):
    """Ensure reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# Use before training
set_random_seeds(42)
trainer = Trainer(hyperparams="default")
model1, dir1 = trainer.train(total_timesteps=25000)

# Reset and train again - should get similar results
set_random_seeds(42)
model2, dir2 = trainer.train(total_timesteps=25000)
```

---

## 📚 API Reference

### Core Classes

#### `bigtwo_rl.training.Trainer`

Main interface for training PPO agents.

```python
Trainer(
    hyperparams="default",           # str or dict: hyperparameter preset or custom config
    reward_function="default",       # str or BaseReward: reward function
    save_every=10000,               # int: checkpoint frequency
    verbose=1,                      # int: logging level (0=silent, 1=progress, 2=detailed)
    controlled_player=0,            # int: player seat controlled by PPO (0-3)
    opponent_mixture=None,          # dict: opponent sampling weights
    snapshot_dir=None,              # str: directory for self-play snapshots
    snapshot_every_steps=10000,     # int: snapshot frequency
    eval_freq=None                  # int: evaluation frequency
)
```

**Methods:**
- `train(total_timesteps, description=None)` → `(model, model_directory)`

#### `bigtwo_rl.evaluation.Evaluator`

Comprehensive model evaluation in 4-player games.

```python
Evaluator(
    num_games=50,                   # int: number of evaluation games
    verbose=True,                   # bool: show progress
    opponent_agents=None,           # list: custom opponents (uses defaults if None)
    n_processes=1                   # int: multiprocessing (experimental)
)
```

**Methods:**
- `evaluate_model(model_path)` → `dict` with keys: `players`, `wins`, `win_rates`, `avg_cards_left`, `cards_left_by_game`

#### `bigtwo_rl.evaluation.Tournament`

Multi-agent tournament system.

```python
Tournament(
    agents,                         # list: agents to compete
    n_processes=None,               # int: parallel processes (None=auto-detect)
    verbose=True                    # bool: detailed output
)
```

**Methods:**
- `run_round_robin(num_games)` → `dict` with tournament results
- `run(num_games)` → Alias for `run_round_robin`

#### Agent Classes

All agents inherit from `bigtwo_rl.agents.BaseAgent`:

- **`RandomAgent(name)`**: Random action selection
- **`GreedyAgent(name)`**: Always plays lowest valid card  
- **`PPOAgent(model_path, name)`**: Trained PPO model wrapper

### Environment Classes

#### `bigtwo_rl.core.BigTwoRLWrapper`

Gymnasium-compatible environment for RL training.

```python
BigTwoRLWrapper(
    num_players=4,                  # int: number of players (always 4)
    games_per_episode=5,            # int: games per training episode
    reward_function=None,           # BaseReward: custom reward function
    controlled_player=0,            # int: player controlled by agent
    opponent_provider=None          # OpponentProvider: custom opponent logic
)
```

**Key Methods:**
- `reset()` → `(observation, info)`
- `step(action)` → `(observation, reward, terminated, truncated, info)`
- **Observation Space**: 109 features (hand + last play + game state)
- **Action Space**: Dynamic based on legal moves

### Utility Functions

```python
# Hyperparameter management
from bigtwo_rl.training.hyperparams import get_config, list_configs

configs = list_configs()  # ['default', 'aggressive', 'conservative', 'fast_experimental']
config = get_config("default")  # Returns dict of hyperparameters

# Card utilities  
from bigtwo_rl.core.card_utils import display_card, parse_card, card_to_unicode

display_card("AS")  # "A♠"
parse_card("KH")    # Returns internal representation
```

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/your-username/bigtwo-rl.git
cd bigtwo-rl

# 2. Install development dependencies
uv sync

# 3. Run tests to ensure everything works
uv run python tests/test_wrapper.py
uv run python tests/test_training.py
uv run python tests/test_evaluation.py

# 4. Install pre-commit hooks (optional)
pre-commit install
```

### Contributing Guidelines

1. **Code Style**: Follow PEP 8, use type hints where helpful
2. **Testing**: Add tests for new functionality in `tests/`
3. **Documentation**: Update docstrings and README for API changes
4. **Performance**: Maintain or improve benchmark performance
5. **Backwards Compatibility**: Avoid breaking existing APIs when possible

### Areas for Contribution

- 🎯 **New Reward Functions**: Different training objectives
- 🤖 **Agent Implementations**: Alternative algorithms (DQN, A3C, etc.)
- 📊 **Visualization Tools**: Training progress, game analysis
- 🏆 **Tournament Features**: New competition formats
- ⚡ **Performance**: Further optimizations and benchmarks
- 🧪 **Research**: Hyperparameter studies, ablation analyses

### Running the Full Test Suite

```bash
# Unit tests
uv run python tests/test_wrapper.py
uv run python tests/test_cards.py
uv run python tests/test_rewards.py
uv run python tests/test_training.py
uv run python tests/test_evaluation.py

# Performance benchmarks
uv run python tests/test_numpy_performance.py
uv run python tests/test_optimization_impact.py

# Integration tests
uv run python examples/train_agent.py    # Should complete in ~3 minutes
uv run python examples/tournament_example.py  # Should show tournament results
```

---

## 📄 License

[Add your license information here]

---

## 📖 Citation

If you use this library in your research, please cite:

```bibtex
@software{bigtwo_rl,
  title={Big Two RL Agent Library: High-Performance Reinforcement Learning for Big Two Card Game},
  author={[Your Name/Organization]},
  year={2024},
  url={https://github.com/your-username/bigtwo-rl},
  note={A comprehensive library for training and evaluating RL agents in Big Two}
}
```

---

## 🙏 Acknowledgments

- Built with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for PPO implementation
- Optimized with [NumPy](https://numpy.org/) for high-performance game logic
- Inspired by Big Two card game research and competitive RL environments

---

**Happy Training! 🎯🤖**

For questions, issues, or contributions, please visit our [GitHub repository](https://github.com/your-username/bigtwo-rl) or open an issue.