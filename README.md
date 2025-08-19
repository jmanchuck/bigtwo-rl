# Big Two RL Agent Library

A comprehensive reinforcement learning library for training AI agents to play Big Two (Chinese card game) using PPO (Proximal Policy Optimization). The library provides a clean, extensible API for researchers and developers to experiment with different training approaches.

## Features

- **High-Performance Core**: Fully vectorized numpy implementation for 5-20x speedup
- **Library Architecture**: Proper Python package with clear module organization
- **Extensible Training**: Configurable hyperparameters and custom reward functions 
- **Agent System**: Modular agent implementations (Random, Greedy, PPO)
- **Tournament Framework**: 4-player series and tournaments with statistics
- **Easy Integration**: Simple API for common workflows
- **Complete Big Two Implementation**: Full game rules with all hand types
- **Memory Efficient**: Optimized data structures using boolean arrays

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

## Quick Start

### Training a PPO Agent
```python
from bigtwo_rl.training import Trainer

# Simple training with defaults
trainer = Trainer()
model, model_dir = trainer.train(total_timesteps=25000)
print(f"Model saved to: {model_dir}")
```

### Evaluating a Trained Agent (4-player series)
```python
from bigtwo_rl.evaluation import Evaluator

evaluator = Evaluator(num_games=100)
results = evaluator.evaluate_model("./models/my_model/best_model")
# results is a series summary dict with keys:
#  - players, wins, win_rates, avg_cards_left, draws, games_played, cards_left_by_game
print("Players:", results["players"]) 
print("Wins:", results["wins"]) 
print("Win rates:", results["win_rates"]) 
print("Avg cards left:", results["avg_cards_left"]) 
```

### Running a Tournament (4-player tables)
```python
from bigtwo_rl.agents import RandomAgent, GreedyAgent, PPOAgent
from bigtwo_rl.evaluation import Tournament

agents = [
    RandomAgent("Random"),
    RandomAgent("Random-2"),
    GreedyAgent("Greedy"),
    PPOAgent("./models/my_model/best_model", "MyAgent"),
]

tournament = Tournament(agents)
results = tournament.run_round_robin(num_games=100)  # plays all 4-agent combinations
print(results["tournament_summary"])
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

### Agent Types
- **RandomAgent**: Random baseline for evaluation
- **GreedyAgent**: Always plays lowest valid card
- **PPOAgent**: Wrapper for trained PPO models

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite: `uv run python tests/`
5. Submit a pull request

## License

[License information]

## Citation

If you use this library in your research, please cite:

```bibtex
@software{bigtwo_rl,
  title={Big Two RL Agent Library},
  author={[Authors]},
  year={2024},
  url={[Repository URL]}
}
```