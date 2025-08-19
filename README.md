# Big Two RL Agent

Train AI agents to play Big Two (Chinese poker) using reinforcement learning! 🃏🤖

## What is This?

This project trains AI agents to play Big Two through reinforcement learning - they learn optimal strategies by playing thousands of games and improving from experience, just like humans do.

**Key Features:**
- 🎛️ **Configurable Training**: Easy hyperparameter and reward function experimentation
- ⚔️ **Agent Tournaments**: Pit different AI models against each other
- 🎯 **Multiple Strategies**: Train agents with different reward structures and learning approaches
- 🎮 **Play vs AI**: Interactive CLI to test your skills against trained agents

## Quick Start

```bash
# Setup
uv sync                                    # Install dependencies

# Train your first agent (fast experimental run)
uv run train_with_config.py fast_experimental default 10000

# Evaluate how well it learned
uv run evaluate.py ./models/fast_experimental_default/best_model

# Play against it!
uv run play_vs_agent.py ./models/fast_experimental_default/best_model
```

## Training Your Own Agents

### Available Configurations

**Hyperparameter Sets:**
- `default`: Balanced learning (recommended for most experiments)
- `aggressive`: Fast learning, higher risk 
- `conservative`: Slow, stable learning
- `fast_experimental`: Quick testing (for development)

**Reward Functions:**
- `default`: Win +5, penalty based on remaining cards
- `sparse`: Simple win/loss (+1/-1)
- `aggressive_penalty`: Harsh penalties for losing badly
- `progressive`: Rewards progress (fewer cards = better)
- `ranking`: Based on final position among all players

### Training Examples

```bash
# List all available options
uv run train_with_config.py --list-configs

# Standard training run
uv run train_with_config.py default default 50000

# Experiment with different reward structures
uv run train_with_config.py default sparse 25000 sparse_experiment
uv run train_with_config.py default progressive 25000 progress_experiment

# Try aggressive learning
uv run train_with_config.py aggressive aggressive_penalty 30000 aggressive_agent
```

### Agent Tournaments

Compare different training approaches:

```bash
# Simple tournament example
uv run python tournament.py example

# Custom tournament (Python script)
python -c "
from tournament import run_round_robin_tournament
from agents import RandomAgent, GreedyAgent, PPOAgent

agents = [
    RandomAgent('Random'),
    GreedyAgent('Greedy'),
    PPOAgent('./models/sparse_experiment/best_model', 'Sparse-Agent'),
    PPOAgent('./models/progress_experiment/best_model', 'Progress-Agent')
]

results = run_round_robin_tournament(agents, num_games=50)
print(results['tournament_summary'])
"
```

## Understanding the AI

### How Reinforcement Learning Works Here

**The Learning Process:**
1. **Observation**: AI sees its hand + game state (last played cards, other players' hand sizes)
2. **Decision**: Neural network picks an action (which cards to play or pass)
3. **Feedback**: Game responds with new state + reward signal
4. **Learning**: AI adjusts to favor actions that led to better outcomes

**Key Concepts:**
- **Multi-Game Episodes**: Instead of learning from single hands (too random), each "episode" consists of multiple full games. The AI gets rewarded based on average performance, helping it learn true skill.
- **Self-Play**: AI trains by playing against copies of itself. As it improves, opponents also improve, creating continuous challenge.
- **Action Masking**: AI can only choose legal moves, focusing learning on strategy rather than rules.

### Monitoring Training

```bash
# View training progress in real-time
tensorboard --logdir=./logs

# Or for specific experiment
tensorboard --logdir=./logs/your_experiment_name
```

### Game Implementation

**Full Big Two Rules Implemented:**
- ✅ Singles, pairs, trips, and all 5-card hands
- ✅ Proper Big Two rankings (2 highest, A second highest)  
- ✅ All hand types: straight, flush, full house, four-of-a-kind, straight flush
- ✅ 2-4 player support with correct hand distributions

**AI Features:**
- 109-dimensional observation space (hand + game state)
- Dynamic action space with legal move masking
- Multi-game episode training for stable skill assessment

## Results & Performance

**Current Achievements:**
- ✅ 100% win rate vs random and greedy baselines after 50k timesteps
- ✅ Episode length reduction from 69→43 steps (learned efficiency)
- ✅ Successful convergence with full Big Two complexity
- ✅ Modular system enables easy comparison of different approaches

**Example Training Results:**
```bash
# Train two different approaches
uv run train_with_config.py default default 50000 standard_agent
uv run train_with_config.py default sparse 50000 sparse_agent

# Compare them in tournament
# (sparse reward often leads to more aggressive play styles)
```

## Testing & Validation

```bash
# Test the environment wrapper
uv run test_wrapper.py

# Test card utilities  
uv run test_cards.py

# Test reward functions
uv run test_rewards.py

# Test training setup
uv run test_training.py
```

## Tips for Experimentation

**Quick Testing:**
- Use `fast_experimental` config for rapid prototyping
- Start with 10k-25k timesteps to test ideas
- Use sparse rewards for simpler learning signals

**Serious Training:**
- Use `default` or `conservative` configs for production models
- Train for 50k+ timesteps for stable performance  
- Try different reward functions to encourage different play styles

**Model Comparison:**
- Always run tournaments between your models
- Test against baselines (random/greedy) to validate learning
- Use evaluation scripts to get quantitative metrics

## Troubleshooting

**Common Issues:**
- **Observation shape errors**: Make sure models were trained with same environment version
- **Slow tournaments**: Reduce number of games for quick testing  
- **Models not improving**: Try different reward functions or hyperparameter sets

**Getting Help:**
- Check the logs in `./logs/` for training progress
- Use tensorboard to visualize learning curves
- Test individual components with the provided test scripts