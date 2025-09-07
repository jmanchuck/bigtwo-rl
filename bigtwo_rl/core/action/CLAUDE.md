# bigtwo_rl/core/action/ Directory

## Overview

The `action/` directory implements the complete 1,365-dimensional action space for Big Two RL. This system provides precise action enumeration, validation, and translation between action IDs and actual card combinations. It's the core of what makes the RL training efficient and accurate.

## Directory Structure

```
action/
├── __init__.py          # Main exports: ActionMaskBuilder, constants, lookup tables
├── constants.py         # Action space layout and offsets (1,365 actions total)
├── engines.py           # BitsetFiveCardEngine for five-card hand classification
├── lookups.py           # Precomputed lookup tables for action-to-tuple conversion
├── mask_builder.py      # ActionMaskBuilder class for generating legal action masks
├── calc.py             # Hand calculation and comparison utilities
├── util.py             # Utility functions for action space manipulation
└── utils.py            # Mathematical utilities (combinations, indexing)
```

## Action Space Architecture

### 1,365-Action Layout
```
Action ID Range    | Type           | Count | Description
0                  | Pass           | 1     | Pass turn (only allowed after first play)
1-13              | Singles        | 13    | Individual cards from hand slots 0-12
14-46             | Pairs          | 33    | All possible pairs from available cards
47-77             | Triples        | 31    | All possible triples from available cards
78-1364           | Five-card      | 1287  | All C(13,5) five-card combinations
```

### Key Constants (`constants.py`)
```python
OFF_PASS = 0       # Pass action index
OFF_1 = 1          # Singles start index
OFF_2 = 14         # Pairs start index  
OFF_3 = 47         # Triples start index
OFF_5 = 78         # Five-card combos start index
N_ACTIONS = 1365   # Total action space size
```

## Core Components

### 1. ActionMaskBuilder (`mask_builder.py`)
**Primary Interface**: Generates legal action masks based on current game state.

**Key Methods:**
```python
def full_mask_indices(self, hand, last_played_cards, pass_allowed, is_first_play, has_control):
    """Returns list of valid action IDs for current game state."""
    
def single_and_multiples(self, hand, last_kind=None, last_key=None):
    """Generate action IDs for singles, pairs, and triples."""
    
def five_card_actions(self, hand, last_five=None):
    """Generate action IDs for five-card combinations."""
```

**Usage:**
```python
from bigtwo_rl.core.action import ActionMaskBuilder, BitsetFiveCardEngine

engine = BitsetFiveCardEngine()
masker = ActionMaskBuilder(engine)

# Get valid actions
valid_actions = masker.full_mask_indices(
    current_hand,           # Hand object with 13 card slots
    last_played_cards,      # LastFive object or None
    pass_allowed=True,      # Can pass this turn
    is_first_play=False,    # Not first play of game
    has_control=True        # Player has control
)
```

### 2. Lookup Tables (`lookups.py`)
**Performance Optimization**: Precomputed mappings for instant action-to-tuple conversion.

**Key Lookup Tables:**
- `PAIR_LUT`: Maps pair types to action IDs
- `TRIPLE_LUT`: Maps triple types to action IDs
- `PAIR_ID`/`TRIPLE_ID`: Reverse mappings from ranks to action IDs
- `SINGLE_REVERSE`/`PAIR_REVERSE`/`TRIPLE_REVERSE`: Action ID to card slot mappings

**Critical Function:**
```python
def action_to_tuple(action_id: int) -> Tuple[int, ...]:
    """Convert action ID to tuple of card slot indices.
    
    Examples:
    - action_to_tuple(0) -> ()  # Pass
    - action_to_tuple(5) -> (4,)  # Single card from slot 4
    - action_to_tuple(20) -> (2, 7)  # Pair from slots 2 and 7
    - action_to_tuple(100) -> (0, 3, 5, 8, 12)  # Five-card combo
    """
```

### 3. Five-Card Engine (`engines.py`)
**Hand Classification**: Determines hand types and strengths for five-card combinations.

**BitsetFiveCardEngine Features:**
- Hand type classification (straight, flush, full house, etc.)
- Hand strength comparison using bitset operations
- Efficient hand validation and ranking

**Usage:**
```python
from bigtwo_rl.core.action import BitsetFiveCardEngine
from bigtwo_rl.core.game import compute_key_and_hand_type

engine = BitsetFiveCardEngine()

# Classify a five-card hand
cards = [card1, card2, card3, card4, card5]  # Encoded cards
key, hand_type = compute_key_and_hand_type(cards)

# Check if hand beats another
can_play = engine.beats(new_hand_key, new_hand_type, last_hand_key, last_hand_type)
```

### 4. Mathematical Utilities (`utils.py`)
**Combination Logic**: Handles mathematical operations for action space indexing.

**Key Functions:**
```python
def comb_index_5(slots: Tuple[int, ...]) -> int:
    """Convert 5-card slot combination to index in C(13,5) space."""
    
def comb_5_from_index(index: int) -> Tuple[int, int, int, int, int]:
    """Convert index back to 5-card slot combination."""
    
def _choose_k(n: int, k: int) -> int:
    """Efficient binomial coefficient calculation."""
```

## Action Translation Pipeline

### From Action ID to Game Move
1. **Input**: Action ID (0-1364) from RL agent
2. **Lookup**: Use `action_to_tuple()` to get card slot indices
3. **Hand Access**: Extract actual cards from Hand object using slot indices
4. **Validation**: Verify cards are available (not played) in hand
5. **Game Format**: Convert to 52-card boolean array for ToyBigTwoFullRules
6. **Execution**: Pass to game engine via `game.step()`

### Example Translation
```python
# Agent selects action 156 (some five-card combo)
action_id = 156

# Convert to slot indices
slots = action_to_tuple(action_id)  # e.g., (1, 4, 7, 9, 11)

# Get actual cards from hand
cards = []
for slot in slots:
    if not hand.played[slot]:  # Verify card available
        cards.append(hand.card[slot])

# Convert to game engine format (52-card boolean array)
game_move = np.zeros(52, dtype=bool)
for card in cards:
    rank = card >> 2
    suit = card & 3
    game_index = rank * 4 + suit  # Convert to legacy format
    game_move[game_index] = True
```

## Action Masking Logic

### Legal Action Determination
The ActionMaskBuilder considers multiple factors to determine valid actions:

1. **Card Availability**: Only cards in hand (not played) can be used
2. **Hand Type Matching**: Must play same type as last play (single, pair, etc.)
3. **Strength Comparison**: New play must beat last play's strength
4. **Special Rules**: 
   - First play must include 3♦
   - Pass only allowed after first play
   - Control player can play any legal combination

### Masking Process
```python
def generate_mask(game_state):
    valid_actions = []
    
    # Always check pass (unless first play)
    if not is_first_play:
        valid_actions.append(OFF_PASS)
    
    # Get last play requirements
    last_type = get_last_hand_type()
    last_strength = get_last_hand_strength()
    
    # Generate actions by type
    if last_type == HandType.SINGLE or last_type is None:
        valid_actions.extend(generate_single_actions(hand, last_strength))
    
    if last_type == HandType.PAIR or last_type is None:
        valid_actions.extend(generate_pair_actions(hand, last_strength))
        
    # ... similar for triples and five-card hands
    
    # Convert to boolean mask
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[valid_actions] = True
    return mask
```

## Performance Optimizations

### 1. Precomputed Lookup Tables
- All action-to-tuple mappings precomputed at module load
- O(1) action translation instead of runtime calculation
- Memory trade-off for speed (essential for RL training)

### 2. Bitset Operations
- Hand classification uses efficient bitset operations
- Rank/suit extraction via bit shifting: `rank = card >> 2`
- Fast comparisons for hand strength

### 3. Optimized Hand Access
- Direct array indexing instead of loops where possible
- Early termination in hand generation
- Cached hand properties (derived fields)

## Integration with Other Systems

### With Observation System
- Action masks inform observation builders about legal moves
- Legal action count included in observation features
- Action space size (1365) hardcoded in observation builders

### With Game Bridge
- GameBridge converts between action system's Hand objects and legacy game format
- Handles card encoding differences (bigtwo-agent vs legacy)
- Translates game state for action masking

### With Training System
- Action masks passed to RL agents for legal move selection
- Stable-baselines3 integration via `env.action_masks()`
- Action space registered as `spaces.Discrete(1365)`

## Usage Examples

### Basic Action Masking
```python
# In environment step
from bigtwo_rl.core.action import ActionMaskBuilder, BitsetFiveCardEngine

# Setup
engine = BitsetFiveCardEngine()
masker = ActionMaskBuilder(engine)

# Get current game state
bridge = create_game_bridge(game)
current_hand = bridge.get_current_player_hand()
last_played = bridge.get_last_played_cards()

# Generate action mask
valid_actions = masker.full_mask_indices(
    current_hand, last_played, 
    pass_allowed=True, is_first_play=False, has_control=True
)

# Convert to boolean array for agent
mask = np.zeros(1365, dtype=bool)
mask[valid_actions] = True
```

### Action Translation
```python
from bigtwo_rl.core.action import action_to_tuple

# Agent selects action
action_id = agent.get_action(obs, action_mask)

# Translate to card slots
if action_id == OFF_PASS:
    slots = ()  # Pass action
else:
    slots = action_to_tuple(action_id)

# Execute in game
game_move = convert_slots_to_game_move(slots, current_hand)
obs, reward, done, info = game.step(game_move)
```

## Development Notes

### Adding New Action Types
To extend the action space:
1. Update `constants.py` with new offsets
2. Add lookup tables in `lookups.py`
3. Extend ActionMaskBuilder with new generation methods
4. Update `action_to_tuple()` function
5. Adjust N_ACTIONS constant

### Performance Considerations
- Action masking happens every environment step
- Must be extremely fast (< 1ms for training efficiency)
- Precompute everything possible at module load
- Use numpy operations over Python loops

### Testing Strategy
- Unit tests for each action type generation
- Integration tests with full game scenarios
- Performance benchmarks for masking operations
- Validation that all 1,365 actions are reachable

## Common Issues

### 1. Action ID Out of Range
```python
# Always validate action IDs
if not (0 <= action_id < N_ACTIONS):
    raise ValueError(f"Action ID {action_id} out of range [0, {N_ACTIONS})")
```

### 2. Invalid Hand State
```python
# Ensure hand has exactly 13 slots
if len(hand.card) != 13 or len(hand.played) != 13:
    raise ValueError("Hand must have exactly 13 slots")
```

### 3. Action Translation Failures
```python
# Handle invalid action IDs gracefully
try:
    slots = action_to_tuple(action_id)
except (IndexError, KeyError):
    # Fall back to pass action
    slots = ()
```

The action system is the most performance-critical component of the RL environment, as it's called on every environment step. The design prioritizes speed through precomputation while maintaining flexibility for different game scenarios.