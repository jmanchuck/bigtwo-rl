# bigtwo_rl/core/cards/ Directory

## Overview

The `cards/` directory implements the modern card encoding system used throughout the bigtwo-agent components. This system uses efficient bit manipulation for card representation and provides the foundation for the action and observation systems. It represents cards in the `(rank<<2)|suit` format optimized for Big Two gameplay.

## Directory Structure

```
cards/
├── __init__.py     # Main exports: constants, encoding functions, conversion utilities
└── cards.py        # Complete card system implementation
```

## Card Encoding System

### Bit-Packed Format
**Core Encoding**: Each card is represented as a single integer using `(rank<<2)|suit`.

```python
# Card encoding: (rank << 2) | suit
# Example: King of Hearts = (10 << 2) | 2 = 42

def encode(rank: int, suit: int) -> int:
    """Encode card as one byte: (rank<<2)|suit."""
    return rank << 2 | suit

def rank_of(card_code: int) -> int:
    """Extract rank from encoded card."""
    return card_code >> 2

def suit_of(card_code: int) -> int:
    """Extract suit from encoded card."""
    return card_code & 3
```

### Big Two Specific Ordering
**Rank Order**: 3(0) < 4(1) < 5(2) < 6(3) < 7(4) < 8(5) < 9(6) < 10(7) < J(8) < Q(9) < K(10) < A(11) < 2(12)

**Suit Order**: ♦(0) < ♣(1) < ♥(2) < ♠(3)

```python
# Constants
RANKS = list(range(13))  # [0, 1, 2, ..., 12]
SUITS = list(range(4))   # [0, 1, 2, 3] 

# All 52 cards in encoded format
ALL_CARDS = [r << 2 | s for r in RANKS for s in SUITS]

# Special cards
THREE_DIAMONDS = 0 << 2 | 0  # rank=0, suit=0, lowest card
```

### String Representation
**Human-Readable Format**: Cards displayed as rank+suit strings like "3D", "AS", "2H".

```python
def card_to_string(card_code: int) -> str:
    """Convert encoded card to string like '3D', 'AS', etc."""
    rank = rank_of(card_code)
    suit = suit_of(card_code)
    
    rank_chars = "3456789TJQKA2"
    suit_chars = "DCHS"  # ♦♣♥♠
    
    return rank_chars[rank] + suit_chars[suit]

def string_to_card(card_str: str) -> int:
    """Parse card string like '3D', 'AS' into encoded card."""
    rank_char = card_str[0]
    suit_char = card_str[1]
    
    rank_map = {
        "3": 0, "4": 1, "5": 2, "6": 3, "7": 4, "8": 5, "9": 6,
        "T": 7, "J": 8, "Q": 9, "K": 10, "A": 11, "2": 12
    }
    suit_map = {"D": 0, "C": 1, "H": 2, "S": 3}
    
    return encode(rank_map[rank_char], suit_map[suit_char])
```

## Core Functions

### Card Creation and Access
```python
from bigtwo_rl.core.cards import encode, rank_of, suit_of

# Create cards
three_diamonds = encode(0, 0)  # 3♦
ace_spades = encode(11, 3)     # A♠  
two_hearts = encode(12, 2)     # 2♥ (highest rank)

# Extract properties
rank = rank_of(ace_spades)  # 11
suit = suit_of(ace_spades)  # 3

# Convert to/from strings
card_str = card_to_string(ace_spades)      # "AS"
card_code = string_to_card("AS")           # 47
```

### All Cards Generation
```python
from bigtwo_rl.core.cards import ALL_CARDS, RANKS, SUITS

# Generate all 52 cards
for rank in RANKS:
    for suit in SUITS:
        card = encode(rank, suit)
        print(f"{card_to_string(card)}: {card}")

# Or use precomputed list
print(f"Total cards: {len(ALL_CARDS)}")  # 52
print(f"First card: {card_to_string(ALL_CARDS[0])}")   # "3D"
print(f"Last card: {card_to_string(ALL_CARDS[-1])}")   # "2S"
```

## Integration with Other Systems

### Action System Integration
The action system uses this encoding for card slot management:

```python
# In ActionMaskBuilder
def _generate_singles(self, hand: Hand, last_kind, last_key):
    for i in range(13):
        if hand.played[i]:
            continue
        card_code = hand.card[i]  # Uses bigtwo-agent encoding
        rank = rank_of(card_code)
        suit = suit_of(card_code)
        # ... action generation logic
```

### Observation System Integration
Observation builders convert encoded cards to feature vectors:

```python
# In BasicObservationBuilder
def encode_hand(self, hand: Hand) -> np.ndarray:
    hand_bits = np.zeros(52, dtype=np.bool_)
    
    for i in range(13):
        if hand.played[i]:
            continue
        card_code = hand.card[i]  # bigtwo-agent format
        rank = rank_of(card_code)
        suit = suit_of(card_code)
        
        # Convert to deck index for observation
        deck_idx = rank * 4 + suit
        hand_bits[deck_idx] = True
    
    return hand_bits
```

### Game Bridge Integration
The game bridge converts between card formats:

```python
# In game_bridge.py
def convert_hand_to_bigtwo_agent(self, player_idx: int) -> Hand:
    # Get legacy format cards
    player_cards = self.game.hands[player_idx]
    card_indices = np.where(player_cards)[0]
    
    hand_cards = []
    for card_idx in card_indices:
        # Convert legacy index to rank/suit
        rank = card_idx // 4
        suit = card_idx % 4
        # Encode in bigtwo-agent format
        encoded_card = encode(rank, suit)
        hand_cards.append(encoded_card)
    
    return Hand(card=hand_cards, played=[0] * len(hand_cards))
```

## Key Differences from Legacy System

### Encoding Comparison

**Legacy System** (`card_utils.py`):
```python
# Uses card indices 0-51
def card_to_string(card_idx: int) -> str:
    rank = card_idx // 4
    suit = card_idx % 4
    # Same rank/suit order, different encoding
```

**Modern System** (`cards/cards.py`):
```python
# Uses bit-packed encoding
def card_to_string(card_code: int) -> str:
    rank = card_code >> 2  # Extract via bit shift
    suit = card_code & 3   # Extract via bit mask
```

### Performance Benefits
1. **Bit Operations**: Faster rank/suit extraction using bit shifts
2. **Memory Efficient**: Single integer per card vs. index lookups
3. **Cache Friendly**: Better memory locality for card operations
4. **Hardware Optimized**: Bit operations are CPU primitive instructions

### Compatibility Layer
The GameBridge handles conversion between systems:

```python
# Legacy to modern
legacy_index = 42  # K♥ in legacy format
rank = legacy_index // 4      # 10
suit = legacy_index % 4       # 2
modern_code = encode(rank, suit)  # 42 in modern format

# Modern to legacy  
modern_code = 42
rank = rank_of(modern_code)   # 10
suit = suit_of(modern_code)   # 2
legacy_index = rank * 4 + suit  # 42 in legacy format
```

## Usage Examples

### Basic Card Operations
```python
from bigtwo_rl.core.cards import *

# Create specific cards
cards = [
    encode(0, 0),   # 3♦ (lowest)
    encode(11, 3),  # A♠
    encode(12, 2),  # 2♥ (highest rank)
]

# Display cards
for card in cards:
    print(f"Card {card}: {card_to_string(card)}")

# Sort cards by Big Two rules
def big_two_key(card):
    rank = rank_of(card)
    suit = suit_of(card)
    return (rank, suit)

sorted_cards = sorted(cards, key=big_two_key)
print("Sorted:", [card_to_string(c) for c in sorted_cards])
```

### Hand Analysis
```python
from bigtwo_rl.core.cards import *

def analyze_hand(hand_cards):
    """Analyze a hand of cards."""
    print(f"Hand size: {len(hand_cards)}")
    
    # Group by rank
    ranks = {}
    for card in hand_cards:
        rank = rank_of(card)
        if rank not in ranks:
            ranks[rank] = []
        ranks[rank].append(card)
    
    # Find pairs, triples, etc.
    for rank, cards in ranks.items():
        rank_name = "3456789TJQKA2"[rank]
        if len(cards) >= 2:
            card_strs = [card_to_string(c) for c in cards]
            print(f"{rank_name}: {card_strs} ({'pair' if len(cards)==2 else 'triple'})")

# Example usage
hand = [encode(r, s) for r, s in [(0,0), (0,1), (5,2), (5,3), (12,0)]]
analyze_hand(hand)
```

### String Parsing
```python
from bigtwo_rl.core.cards import string_to_card, card_to_string

def parse_card_list(card_string):
    """Parse space-separated card string into encoded cards."""
    card_strs = card_string.upper().split()
    cards = []
    
    for card_str in card_strs:
        try:
            card = string_to_card(card_str)
            cards.append(card)
        except KeyError:
            print(f"Invalid card: {card_str}")
    
    return cards

# Example
user_input = "3D AS 2H KS"
cards = parse_card_list(user_input)
print("Parsed cards:", [card_to_string(c) for c in cards])
```

## Performance Characteristics

### Bit Operation Efficiency
```python
import timeit

# Encoding performance
def test_encode():
    return encode(10, 2)  # K♥

def test_decode():
    card = 42
    return rank_of(card), suit_of(card)

# These operations are extremely fast (~10ns each)
encode_time = timeit.timeit(test_encode, number=1000000)
decode_time = timeit.timeit(test_decode, number=1000000)

print(f"Encode time: {encode_time:.6f}s for 1M operations")
print(f"Decode time: {decode_time:.6f}s for 1M operations")
```

### Memory Usage
```python
# Memory comparison
import sys

# Legacy: store as list of indices
legacy_hand = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
legacy_memory = sys.getsizeof(legacy_hand) + sum(sys.getsizeof(x) for x in legacy_hand)

# Modern: store as list of encoded cards
modern_hand = [encode(r, s) for r in range(11) for s in [0]][:11]
modern_memory = sys.getsizeof(modern_hand) + sum(sys.getsizeof(x) for x in modern_hand)

print(f"Legacy memory: {legacy_memory} bytes")
print(f"Modern memory: {modern_memory} bytes")
print(f"Memory ratio: {modern_memory/legacy_memory:.2f}")
```

## Common Patterns

### Card Comparison
```python
def card_beats(card1, card2):
    """Check if card1 beats card2 in Big Two."""
    rank1, suit1 = rank_of(card1), suit_of(card1)
    rank2, suit2 = rank_of(card2), suit_of(card2)
    
    # Compare by rank first, then suit
    if rank1 != rank2:
        return rank1 > rank2
    return suit1 > suit2

# Example
three_d = encode(0, 0)  # 3♦
three_h = encode(0, 2)  # 3♥
print(card_beats(three_h, three_d))  # True (higher suit)
```

### Hand Sorting
```python
def sort_hand(cards):
    """Sort cards by Big Two rules (rank, then suit)."""
    return sorted(cards, key=lambda c: (rank_of(c), suit_of(c)))

# Example
hand = [encode(12, 0), encode(0, 3), encode(5, 1), encode(0, 0)]
sorted_hand = sort_hand(hand)
print("Hand:", [card_to_string(c) for c in sorted_hand])
# Output: ['3D', '3S', '6C', '2D']
```

### Rank/Suit Filtering
```python
def filter_by_rank(cards, target_rank):
    """Get all cards of specific rank."""
    return [c for c in cards if rank_of(c) == target_rank]

def filter_by_suit(cards, target_suit):
    """Get all cards of specific suit."""
    return [c for c in cards if suit_of(c) == target_suit]

# Example
all_aces = filter_by_rank(ALL_CARDS, 11)  # All aces
all_spades = filter_by_suit(ALL_CARDS, 3)  # All spades
print("Aces:", [card_to_string(c) for c in all_aces])
print("Spades:", [card_to_string(c) for c in all_spades])
```

## Future Improvements

### 1. Extended Card Properties
```python
# Potential enhancements
def card_color(card_code):
    """Get card color (red/black)."""
    suit = suit_of(card_code)
    return "red" if suit in [0, 2] else "black"  # ♦♥ are red

def card_value(card_code):
    """Get card value for scoring."""
    rank = rank_of(card_code)
    return 1 if rank == 11 else (2 if rank == 12 else rank + 3)
```

### 2. Validation and Safety
```python
def validate_card(card_code):
    """Validate encoded card is within valid range."""
    if not (0 <= card_code <= 51):
        raise ValueError(f"Invalid card code: {card_code}")
    
    rank = rank_of(card_code)
    suit = suit_of(card_code)
    
    if not (0 <= rank <= 12):
        raise ValueError(f"Invalid rank: {rank}")
    if not (0 <= suit <= 3):
        raise ValueError(f"Invalid suit: {suit}")
    
    return True
```

### 3. Unicode Display
```python
def card_to_unicode(card_code):
    """Convert card to Unicode symbols."""
    rank = rank_of(card_code)
    suit = suit_of(card_code)
    
    rank_chars = "3456789TJQKA2"
    suit_chars = "♦♣♥♠"
    
    return rank_chars[rank] + suit_chars[suit]

# Example: "A♠" instead of "AS"
```

The card system provides the foundation for all game operations and is optimized for both performance and clarity. The bit-packed encoding enables efficient operations while maintaining human readability through string conversion functions.