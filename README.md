# Big Two RL Agent Library

A comprehensive reinforcement learning library for training AI agents to play Big Two (Chinese card game). Features a **fixed 1,365-action space** with proper action masking and flexible observation systems using code from the bigtwo-agent library.

## 🚀 Quick Start

### Installation
```bash
# Development install
git clone <repo-url>
cd bigtwo-rl
uv sync
```

### Test Basic Environment
```python
from bigtwo_rl.core.bigtwo_wrapper import BigTwoWrapper
from bigtwo_rl.training.rewards import ZeroSumReward

# Create environment with fixed 1,365-action space
wrapper = BigTwoWrapper(reward_function=ZeroSumReward())
obs, info = wrapper.reset()
print(f"✓ Environment works - observation shape: {obs.shape}")
print(f"✓ Action space: {wrapper.action_space}")
print(f"✓ Legal actions: {wrapper.get_action_mask().sum()}")

# Training pipeline needs to be rebuilt with new action/observation systems
```

### Test Random Agent
```python
from bigtwo_rl.agents import RandomAgent

# Create a random agent that works with the 1,365-action space
agent = RandomAgent("TestRandom")
action = agent.get_action(obs, wrapper.get_action_mask())
print(f"✓ Random agent selected action: {action}")

# Note: Training system, evaluator, and tournaments need to be rebuilt
# for the new action/observation system
```

## 🔧 Architecture Overview

### Fixed 1,365-Action Space

This library now uses a **fixed action space** of exactly 1,365 actions representing all possible Big Two moves:
- **Singles**: 13 actions (one per rank)  
- **Pairs**: 33 actions (all suit combinations per rank)
- **Triples**: 31 actions (all suit combinations per rank)
- **5-card hands**: 1,287 actions (all rank combinations)
- **Pass**: 1 action
- **Total**: 1,365 discrete actions

### New Action & Observation System

The library has been migrated to use code from the **bigtwo-agent** library, providing:
- **Proper action masking**: Only legal moves are available to agents
- **Efficient lookup tables**: Fast action to move translation  
- **Modular observations**: BasicObservationBuilder with 168 features (52+52+64)
- **Bridge pattern**: Converts between old game engine and new action/observation format

## 📊 Core Components

### BigTwoWrapper Environment
```python
from bigtwo_rl.core.bigtwo_wrapper import BigTwoWrapper
from bigtwo_rl.training.rewards import DefaultReward

# Gymnasium-compatible environment with fixed action space
env = BigTwoWrapper(
    reward_function=DefaultReward(),
    num_players=4,
    games_per_episode=10
)

# Standard RL interface
obs, info = env.reset()
action_mask = env.get_action_mask()  # 1,365-dim boolean array
action = agent.get_action(obs, action_mask)  
next_obs, reward, terminated, truncated, info = env.step(action)
```

### Action System
```python
from bigtwo_rl.core.action import ActionMaskBuilder, BitsetFiveCardEngine

# Create action masking system
five_engine = BitsetFiveCardEngine()
action_masker = ActionMaskBuilder(five_engine)

# Get legal actions for current game state
valid_action_ids = action_masker.full_mask_indices(
    current_hand=hand,          # Cards in player's hand
    last_played_cards=cards,    # Last played cards (or None)
    pass_allowed=True,          # Can player pass?
    is_first_play=False,        # First play of game?
    has_control=False           # Does player control the trick?
)
```

### Observation System
```python
from bigtwo_rl.core.observation import BasicObservationBuilder

# Create observation builder (168 features)
obs_builder = BasicObservationBuilder()

# Build observation vector for current player
obs = obs_builder.build_observation(
    hand=player_hand,                    # Player's cards 
    current_player=player_idx,           # Current player index
    player_card_counts=[13,12,11,10],    # Cards left per player
    last_played_cards=played_cards,      # Last trick cards
    passes=2,                            # Consecutive passes
    is_first_play=False                  # First play flag
)
# Returns: np.ndarray of shape (168,) with dtype float32
```

### Available Agents
```python
from bigtwo_rl.agents import RandomAgent, BaseAgent

# Built-in agents
random_agent = RandomAgent("Random")

# Custom agent example
class MyAgent(BaseAgent):
    def get_action(self, observation, action_mask=None):
        # Select from legal actions only
        legal_actions = np.where(action_mask)[0]
        return legal_actions[0]  # Play first legal action
    
    def reset(self):
        pass  # Reset internal state
```

### Reward Functions
```python
from bigtwo_rl.training.rewards import *

# Available reward functions
DefaultReward()           # Balanced win/loss with card penalties
SparseReward()           # Simple win/loss only  
ZeroSumReward()          # Zero-sum competitive rewards
ProgressiveReward()      # Rewards progress (fewer cards)
StrategicReward()        # Complex strategic bonuses
```

## 🔄 Migration Status

### ✅ Completed
- **Action space**: Migrated to fixed 1,365-action space with proper masking
- **Observations**: Migrated to BasicObservationBuilder (168 features)
- **Environment**: BigTwoWrapper updated to use new systems
- **Random agent**: Compatible with new action space
- **Game bridge**: Conversion between old game engine and new formats

### ⚠️ In Progress  
- **Action translation**: Currently using placeholder pass actions
- **Training system**: Needs rebuild for new action/observation format
- **PPO agent**: Needs update for new systems
- **Evaluator/Tournament**: Need rebuild for new architecture

### 🎯 Next Steps
1. **Implement proper action ID to game move translation**
2. **Rebuild training pipeline with new action/observation systems**  
3. **Update PPOAgent and other agents**
4. **Rebuild evaluation and tournament systems**

## 🚀 Development Commands

```bash
# Environment setup
uv sync

# Test basic functionality
uv run python simple_run.py

# Test individual components (when available)
# uv run python tests/test_wrapper.py
# uv run python tests/test_training.py
```

## 🎮 Game Rules (Big Two)

- **Suits**: ♦ < ♣ < ♥ < ♠ (Diamonds lowest, Spades highest)
- **Ranks**: 3 < 4 < 5 < 6 < 7 < 8 < 9 < 10 < J < Q < K < A < 2 (2 is highest)  
- **Goal**: First player to play all cards wins
- **Hands**: Singles, pairs, trips, 5-card combinations (straights, flushes, etc.)
- **Card encoding**: `(rank << 2) | suit` where rank 0-12, suit 0-3

## 🔧 Technical Details

### Action Space Breakdown
```
Action ID Range    | Type           | Count | Description
0-12              | Singles        | 13    | One card per rank (3-2)
13-169           | Pairs          | 156   | All rank+suit combinations  
170-375          | Triples        | 206   | All rank+suit combinations
376-1363         | 5-card hands   | 988   | All 5-card combinations
1364             | Pass           | 1     | Pass action
Total: 1,365 actions
```

### Observation Features (168 total)
- **Hand features (52)**: Binary presence of each card in player's hand
- **Last play features (52)**: Binary representation of last played cards
- **Game state features (64)**: Card counts + game flags (current player, passes, first play)

---

**Status**: Core environment and action masking working. Training system needs rebuild for new architecture.