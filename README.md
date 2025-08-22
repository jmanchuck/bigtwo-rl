# Big Two RL Agent Library

A comprehensive reinforcement learning library for training AI agents to play Big Two (Chinese card game) using PPO. Train agents with different strategies, compete them in tournaments, and experiment with configurable observations and rewards.

## 🚀 Quick Start

### Installation
```bash
# Development install
git clone <repo-url>
cd bigtwo-rl
uv sync
```

### Train Your First Agent (3 minutes)
```python
from bigtwo_rl.training import Trainer
from bigtwo_rl.training.rewards import DefaultReward
from bigtwo_rl.training.hyperparams import DefaultConfig
from bigtwo_rl import standard_observation

trainer = Trainer(
    reward_function=DefaultReward(),
    hyperparams=DefaultConfig(),
    observation_config=standard_observation()
)
model, model_dir = trainer.train(total_timesteps=25000)
# Model saved to ./models/[timestamp]/
```

### Evaluate Your Agent
```python
from bigtwo_rl.evaluation import Evaluator

evaluator = Evaluator(num_games=100)
results = evaluator.evaluate_model(f"{model_dir}/best_model")
print(f"Win rate: {results['win_rates'][0]:.1%}")
```

### Run Tournaments
```python
from bigtwo_rl.agents import RandomAgent, GreedyAgent, PPOAgent
from bigtwo_rl.evaluation import Tournament

agents = [
    PPOAgent(f"{model_dir}/best_model", "MyAgent"),
    GreedyAgent("Greedy"),
    RandomAgent("Random-1"),
    RandomAgent("Random-2")
]

tournament = Tournament(agents)
results = tournament.run(num_games=200)
print(results["tournament_summary"])
```

## 📖 Complete API Guide

### 🎯 Training Agents

#### Basic Training
```python
from bigtwo_rl.training import Trainer
from bigtwo_rl.training.rewards import DefaultReward
from bigtwo_rl.training.hyperparams import DefaultConfig
from bigtwo_rl import standard_observation

trainer = Trainer(
    reward_function=DefaultReward(),
    hyperparams=DefaultConfig(),
    observation_config=standard_observation()
)
model, model_dir = trainer.train(total_timesteps=25000)
```

#### Available Reward Functions
```python
from bigtwo_rl.training.rewards import *

DefaultReward()           # Balanced win/loss with card penalties
SparseReward()           # Simple win/loss only
ProgressiveReward()      # Rewards progress (fewer cards)
RankingReward()          # Rewards based on final ranking
ScoreMarginReward()      # Continuous reward based on card advantage
StrategicReward()        # Encourages sophisticated play patterns
AggressivePenaltyReward() # Higher penalties for poor performance
ComplexMoveReward(five_card_bonus=0.1) # Bonuses for complex combinations
```

#### Available Hyperparameter Configs
```python
from bigtwo_rl.training.hyperparams import *

DefaultConfig()          # Balanced settings (3 min/25k timesteps)
AggressiveConfig()       # Faster learning (2 min/25k timesteps)
ConservativeConfig()     # Stable training (6 min/25k timesteps)
FastExperimentalConfig() # Quick experiments (1 min/25k timesteps)
```

#### Custom Reward Functions
```python
from bigtwo_rl.training.rewards import BaseReward

class MyReward(BaseReward):
    def game_reward(self, winner_player, player_idx, cards_left, all_cards_left=None):
        if player_idx == winner_player:
            return 10.0
        return -(cards_left ** 2) * 0.5
    
    def episode_bonus(self, games_won, total_games, avg_cards_left):
        return 0  # No episode bonus
    
    def move_bonus(self, move_cards):
        # Optional: bonus for specific move types
        if len(move_cards) == 5:  # 5-card hands
            return 0.2
        return 0

trainer = Trainer(
    reward_function=MyReward(),
    hyperparams=AggressiveConfig(),
    observation_config=standard_observation()
)
```

### 🧠 Observation Configurations

Control what information your agent can see:

#### Pre-built Configurations
```python
from bigtwo_rl import (
    minimal_observation,      # 57 features (hand + hand sizes)
    standard_observation,     # 109 features (+ last play info)
    memory_enhanced_observation, # 217 features (+ card memory)
    strategic_observation     # 300+ features (+ opponent modeling)
)

# Use in training
trainer = Trainer(
    reward_function=DefaultReward(),
    hyperparams=DefaultConfig(),
    observation_config=strategic_observation()  # Rich observations
)
```

#### Custom Observation Configurations
```python
from bigtwo_rl.core.observation_builder import ObservationConfig

# Manual configuration
custom_obs = ObservationConfig(
    include_hand=True,                    # Own cards (required)
    include_last_play=True,               # Current trick cards
    include_hand_sizes=True,              # All players' card counts
    include_played_cards=True,            # Memory of all played cards
    include_power_cards_remaining=True,   # Track 2s and Aces
    include_pass_history=True,            # Who passed on current trick
    include_game_phase=True               # Early/mid/late game indicator
)

trainer = Trainer(
    reward_function=DefaultReward(),
    hyperparams=DefaultConfig(),
    observation_config=custom_obs
)
```

### 📊 Evaluation & Competition

#### Model Evaluation
```python
from bigtwo_rl.evaluation import Evaluator

# Basic evaluation
evaluator = Evaluator(num_games=100)
results = evaluator.evaluate_model("./models/my_model/best_model")

# Access results
win_rate = results['win_rates'][0]
avg_cards_left = results['avg_cards_left'][0] 
total_wins = results['wins'][0]
game_history = results['cards_left_by_game'][0]
```

#### Tournament Competition
```python
from bigtwo_rl.evaluation import Tournament
from bigtwo_rl.agents import RandomAgent, GreedyAgent, PPOAgent

# Set up tournament
agents = [
    PPOAgent("./models/agent1/best_model", "Agent1"),
    PPOAgent("./models/agent2/best_model", "Agent2"),
    GreedyAgent("Greedy"),
    RandomAgent("Random")
]

# Run tournament with multiprocessing
tournament = Tournament(agents, n_processes=None)  # Auto-detect CPUs
results = tournament.run(num_games=500)

# Access results
print(results["tournament_summary"])      # Formatted table
agent_names = results["agents"]           # Agent names
total_wins = results["total_wins"]        # Wins per agent
win_rates = [w/results["total_games"] for w in total_wins]
```

### 🤖 Agent Types

#### Built-in Agents
```python
from bigtwo_rl.agents import RandomAgent, GreedyAgent, PPOAgent

RandomAgent("Random")                                    # Random moves
GreedyAgent("Greedy")                                    # Always lowest card
PPOAgent("./models/my_model/best_model", "MyAgent")      # Trained model
```

#### Custom Agents
```python
from bigtwo_rl.agents import BaseAgent
import numpy as np

class MyCustomAgent(BaseAgent):
    def get_action(self, observation, action_mask=None):
        # Your logic here
        legal_actions = np.where(action_mask)[0]
        return legal_actions[0]  # Play lowest legal action
    
    def reset(self):
        pass  # Reset any internal state
```

### 🎮 Interactive Play

```bash
# Play against a trained agent
uv run python examples/play_vs_agent.py ./models/my_model/best_model
```

### 📈 Training Monitoring

#### TensorBoard Logs
```bash
# Start TensorBoard (in separate terminal)
uv run python -m tensorboard.main --logdir=./logs
# Visit: http://localhost:6006
```

#### Training Progress
```python
trainer = Trainer(
    reward_function=DefaultReward(),
    hyperparams=DefaultConfig(),
    observation_config=standard_observation(),
    eval_freq=5000,    # Evaluate every 5k steps
    verbose=1          # Show progress
)
```

## 🔧 Configuration Reference

### Hyperparameters Explained

Understanding what each setting means in Big Two terms:

- **`learning_rate`**: How fast the AI learns (3e-4 = balanced, 1e-3 = aggressive)
- **`games_per_episode`**: How many complete Big Two games before getting feedback (5 = default)
- **`n_steps`**: Individual card plays before updating AI (512 = ~10-25 games worth)
- **`n_envs`**: Parallel game tables (8 = like 8 different tables at once)
- **`total_timesteps`**: Total card plays for entire training (25k ≈ 835 games)

### Reward Function Guide

| Reward Function | Best For | Description |
|----------------|----------|-------------|
| `DefaultReward` | General training | Balanced win/loss with card penalties |
| `SparseReward` | Simple baseline | Just win (+1) vs loss (-1) |
| `ProgressiveReward` | Aggressive play | Rewards reducing cards each game |
| `StrategicReward` | Advanced play | Encourages sophisticated strategies |
| `ComplexMoveReward` | 5-card hands | Bonuses for complex combinations |
| `RankingReward` | Competitive play | Rewards beating other players |

### Observation Configurations

| Config | Features | Training Speed | Best For |
|--------|----------|----------------|----------|
| `minimal_observation` | 57 | Fastest | Quick experiments, baselines |
| `standard_observation` | 109 | Fast | General training |
| `memory_enhanced_observation` | 217 | Moderate | Card counting strategies |
| `strategic_observation` | 300+ | Slower | Advanced strategic play |

## 🎯 Common Workflows

### Experiment with Different Strategies
```python
# Train agents with different approaches
configs = [
    (DefaultReward(), "default"),
    (ProgressiveReward(), "progressive"), 
    (StrategicReward(), "strategic"),
    (ComplexMoveReward(five_card_bonus=0.2), "complex")
]

agents = []
for reward_fn, name in configs:
    trainer = Trainer(
        reward_function=reward_fn,
        hyperparams=DefaultConfig(),
        observation_config=standard_observation()
    )
    model, model_dir = trainer.train(25000, model_name=f"{name}_agent")
    agents.append(PPOAgent(f"{model_dir}/best_model", name))

# Tournament to see which strategy wins
tournament = Tournament(agents)
results = tournament.run(num_games=200)
print(results["tournament_summary"])
```

### Test Information Advantage
```python
# Does card memory help?
minimal_config = minimal_observation()
memory_config = memory_enhanced_observation()

# Train minimal agent
trainer1 = Trainer(
    reward_function=DefaultReward(),
    hyperparams=DefaultConfig(),
    observation_config=minimal_config
)
model1, dir1 = trainer1.train(25000, model_name="minimal_agent")

# Train memory agent
trainer2 = Trainer(
    reward_function=DefaultReward(),
    hyperparams=DefaultConfig(),
    observation_config=memory_config
)
model2, dir2 = trainer2.train(25000, model_name="memory_agent")

# Head-to-head comparison
agents = [
    PPOAgent(f"{dir1}/best_model", "Minimal", observation_config=minimal_config),
    PPOAgent(f"{dir2}/best_model", "Memory", observation_config=memory_config),
    GreedyAgent("Greedy"),
    RandomAgent("Random")
]

tournament = Tournament(agents)
results = tournament.run(num_games=300)
print("Does memory help?", results["tournament_summary"])
```

## 🚀 Development Commands

```bash
# Environment setup
uv sync

# Test the library
uv run python tests/test_wrapper.py
uv run python tests/test_training.py

# Example workflows  
uv run python examples/strategic_training_example.py
uv run python examples/play_vs_agent.py MODEL
uv run python examples/benchmark_multiprocessing.py

# Monitor training
uv run python -m tensorboard.main --logdir=./logs
```

## 🎮 Game Rules (Big Two)

- **Suits**: ♦ < ♣ < ♥ < ♠ (Diamonds lowest, Spades highest)
- **Ranks**: 3 < 4 < 5 < 6 < 7 < 8 < 9 < 10 < J < Q < K < A < 2 (2 is highest)
- **Goal**: First player to play all cards wins
- **Hands**: Singles, pairs, trips, 5-card combinations (straights, flushes, etc.)

## 📚 Examples & Tutorials

See `/examples/` directory for complete working examples:
- `strategic_training_example.py` - Advanced training configuration
- `play_vs_agent.py` - Interactive play against trained agents
- `benchmark_multiprocessing.py` - Performance testing

---

**Ready to train your Big Two champion? Start with the Quick Start guide above! 🎯**