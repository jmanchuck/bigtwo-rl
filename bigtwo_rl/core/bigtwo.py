import random
import numpy as np
from itertools import combinations
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple, Union, Generator


class ToyBigTwoFullRules:
    """Big Two game engine with complete rule implementation and vectorized operations."""

    # Instance variable type hints
    num_players: int
    hands: np.ndarray  # shape (num_players, 52) boolean array
    current_player: int
    last_play: Optional[Tuple[np.ndarray, int]]  # (cards_mask, player_idx)
    passes_in_row: int
    done: bool

    # Precomputed lookup tables for card properties (massive speedup)
    _RANK_TABLE = [card // 4 for card in range(52)]  # 0 = 3, 12 = 2
    _SUIT_TABLE = [card % 4 for card in range(52)]  # 0 = diamonds, 1 = clubs, 2 = hearts, 3 = spades

    # Precomputed Big Two card values (2 is highest = 12, A = 11, etc.)
    _VALUE_TABLE = []
    _RANK_TO_VALUE = []  # Direct rank-to-value lookup
    _CARD_STRENGTHS = None  # value*4 + suit for each card for fast argmax
    _RANK_MASKS = None  # shape (13, 52) uint8
    _SUIT_MASKS = None  # shape (4, 52) uint8

    for card in range(52):
        rank = card // 4
        if rank == 0:  # 3
            _VALUE_TABLE.append(0)
        elif rank == 1:  # 4
            _VALUE_TABLE.append(1)
        elif rank == 2:  # 5
            _VALUE_TABLE.append(2)
        elif rank == 3:  # 6
            _VALUE_TABLE.append(3)
        elif rank == 4:  # 7
            _VALUE_TABLE.append(4)
        elif rank == 5:  # 8
            _VALUE_TABLE.append(5)
        elif rank == 6:  # 9
            _VALUE_TABLE.append(6)
        elif rank == 7:  # 10
            _VALUE_TABLE.append(7)
        elif rank == 8:  # J
            _VALUE_TABLE.append(8)
        elif rank == 9:  # Q
            _VALUE_TABLE.append(9)
        elif rank == 10:  # K
            _VALUE_TABLE.append(10)
        elif rank == 11:  # A
            _VALUE_TABLE.append(11)
        elif rank == 12:  # 2
            _VALUE_TABLE.append(12)

    # Fill rank-to-value lookup (ranks 0-12 -> values 0-12)
    for rank in range(13):
        _RANK_TO_VALUE.append(rank)  # Since our rank mapping already matches Big Two values

    # Precompute card strengths (value*4 + suit)
    _CARD_STRENGTHS = []
    for card in range(52):
        _CARD_STRENGTHS.append((_VALUE_TABLE[card] * 4) + _SUIT_TABLE[card])

    # Precompute rank and suit masks for vectorized boolean ops
    _RANK_MASKS = [[0] * 52 for _ in range(13)]
    _SUIT_MASKS = [[0] * 52 for _ in range(4)]
    for card in range(52):
        r = card // 4
        s = card % 4
        _RANK_MASKS[r][card] = 1
        _SUIT_MASKS[s][card] = 1
    # Convert to numpy arrays once
    _RANK_MASKS = np.array(_RANK_MASKS, dtype=np.uint8)
    _SUIT_MASKS = np.array(_SUIT_MASKS, dtype=np.uint8)

    def __init__(self, num_players: int = 4, track_move_history: bool = False) -> None:
        self.num_players = num_players
        self.track_move_history = track_move_history
        self.reset()

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed:
            random.seed(seed)
        # Deal cards
        deck = np.arange(52)  # 0..51 = 13 ranks × 4 suits
        np.random.shuffle(deck)

        # Initialize hands as boolean numpy arrays
        self.hands = np.zeros((self.num_players, 52), dtype=bool)
        for i in range(self.num_players):
            player_cards = deck[i :: self.num_players]
            self.hands[i, player_cards] = True

        # Initialize move history tracking (last 5 moves) - only if enabled
        if self.track_move_history:
            self.move_history = []  # List of (cards_mask, player_idx, move_type) tuples

        # Find player with 3 of diamonds (card index 0) to start the game
        three_of_diamonds = 0  # rank 0 (3) * 4 + suit 0 (diamonds) = 0
        starting_player = 0
        for player in range(self.num_players):
            if self.hands[player, three_of_diamonds]:
                starting_player = player
                break

        # Game state
        self.current_player = starting_player
        self.last_play = None  # (cards_array, player)
        self.passes_in_row = 0
        self.done = False
        return self._get_obs()

    def _rank(self, card: int) -> int:
        return self._RANK_TABLE[card]

    def _suit(self, card: int) -> int:
        return self._SUIT_TABLE[card]

    def _card_value(self, card: int) -> int:
        """Get Big Two card value where 2 is highest (12), A is second highest (11), etc."""
        return self._VALUE_TABLE[card]

    @lru_cache(maxsize=50000)
    def _identify_hand_type_cached(self, cards_tuple: Tuple[int, ...]) -> Tuple[str, int]:
        """Cached hand type identification using LRU cache."""
        return self._identify_hand_type_uncached(list(cards_tuple))

    def _identify_hand_type(self, cards: Union[List[int], np.ndarray]) -> Tuple[str, int]:
        """Identify the type of hand and return (hand_type, strength_value)."""
        # Use LRU cache for frequently computed hand types
        # Normalize input to list of indices
        if isinstance(cards, np.ndarray):
            if cards.dtype == bool or len(cards) == 52:
                # Boolean mask - extract indices where True
                cards = np.where(cards)[0].tolist()
            else:
                # Array of card indices - convert to list
                cards = cards.tolist()
        cards_tuple = tuple(sorted(cards))
        return self._identify_hand_type_cached(cards_tuple)

    def _identify_hand_type_uncached(self, cards: List[int]) -> Tuple[str, int]:
        """Actual hand type identification using vectorized operations."""
        if len(cards) == 1:
            return "single", self._card_value(cards[0]) * 4 + self._suit(cards[0])
        elif len(cards) == 2:
            # Vectorized rank and suit extraction
            cards_arr = np.array(cards)
            ranks = cards_arr // 4
            suits = cards_arr % 4
            if ranks[0] == ranks[1]:
                return "pair", self._card_value(cards[0]) * 4 + max(suits)
            else:
                return "invalid", 0
        elif len(cards) == 3:
            # Vectorized rank check
            cards_arr = np.array(cards)
            ranks = cards_arr // 4
            suits = cards_arr % 4
            if len(np.unique(ranks)) == 1:  # All same rank
                return "trips", self._card_value(cards[0]) * 4 + max(suits)
            else:
                return "invalid", 0
        elif len(cards) == 5:
            return self._identify_five_card_hand(cards)
        else:
            return "invalid", 0

    def _identify_five_card_hand(self, cards: List[int]) -> Tuple[str, int]:
        """Identify 5-card hand types using vectorized NumPy mask operations."""
        # Build 52-length boolean mask for the cards
        mask = np.zeros(52, dtype=np.uint8)
        mask[np.array(cards, dtype=np.int64)] = 1

        # Rank and suit counts via precomputed masks
        rank_counts = (self._RANK_MASKS @ mask).astype(np.int8)  # shape (13,)
        suit_counts = (self._SUIT_MASKS @ mask).astype(np.int8)  # shape (4,)
        rank_presence = rank_counts > 0

        # Flush detection
        is_flush = np.any(suit_counts == 5)

        # Straight detection via sliding window on rank presence
        # Convert to Big Two values for straight checking: our ranks already map 0..12 (3..2) and
        # _is_straight previously handled A-2-3-4-5; replicate that rule here on presence.
        presence_int = rank_presence.astype(np.int8)
        # regular straights: any 5 consecutive ranks present
        if presence_int.shape[0] == 13:
            window = np.ones(5, dtype=np.int8)
            window_sums = np.convolve(presence_int, window, mode="valid")  # length 9
            has_regular_straight = np.any(window_sums == 5)
        else:
            has_regular_straight = False
        # A-2-3-4-5 straight special-case
        ace_low_straight = np.sum(presence_int[[0, 1, 2, 11, 12]]) == 5
        is_straight = bool(has_regular_straight or ace_low_straight)

        # Highest card strength helper using precomputed strengths
        strengths = np.array(self._CARD_STRENGTHS, dtype=np.int32)
        # mask is 0/1; multiply to zero-out other cards
        masked_strengths = strengths * mask
        highest_card_strength = int(masked_strengths.max())

        # Check for straight flush
        if is_straight and is_flush:
            return "straight_flush", 50000 + highest_card_strength

        # Check for four of a kind
        if np.any(rank_counts == 4):
            four_rank = int(np.argmax(rank_counts == 4))
            return "four_of_a_kind", 40000 + self._card_value_from_rank(four_rank) * 4

        # Check for full house
        if (np.any(rank_counts == 3)) and (np.any(rank_counts == 2)):
            three_rank = int(np.argmax(rank_counts == 3))
            return "full_house", 30000 + self._card_value_from_rank(three_rank) * 4

        # Check for flush
        if is_flush:
            return "flush", 20000 + highest_card_strength

        # Check for straight
        if is_straight:
            return "straight", 10000 + highest_card_strength

        return "invalid", 0

    def _card_value_from_rank(self, rank: int) -> int:
        """Convert rank (0-12) to Big Two value (0-12)."""
        return self._RANK_TO_VALUE[rank]

    def _is_straight(self, ranks: List[int]) -> bool:
        """Check if ranks form a valid Big Two straight using vectorized operations."""
        # Convert to Big Two values for straight checking
        values = np.array([self._card_value_from_rank(r) for r in ranks])
        values = np.sort(values)

        # Check for regular straight using vectorized operations
        diffs = np.diff(values)
        if np.all(diffs == 1):
            return True

        # Check for A-2-3-4-5 straight (values would be [0,1,2,11,12])
        ace_low_straight = np.array([0, 1, 2, 11, 12])
        if np.array_equal(values, ace_low_straight):
            return True

        return False

    def legal_moves(self, player: int) -> List[np.ndarray]:
        """Generate all legal singles, pairs, trips, and 5-card hands."""
        hand_array = self.hands[player]  # boolean array
        hand_cards = np.where(hand_array)[0]  # get actual card indices
        moves = []

        # Special case: Start of game - first player must play hand containing 3♦.
        # Only enforce this on the very first move of the game, not on each new trick.
        three_of_diamonds = 0  # card index 0 = 3♦
        is_game_start = self.last_play is None and self.passes_in_row == 0 and np.all(self.hands.sum(axis=1) == 13)

        # Singles
        for c in hand_cards:
            single_mask = np.zeros(52, dtype=bool)
            single_mask[c] = True

            # Check game start requirement
            if is_game_start and three_of_diamonds not in hand_cards:
                continue  # This player shouldn't be playing (wrong starting player)
            if is_game_start and not single_mask[three_of_diamonds]:
                continue  # Must include 3♦ in starting play

            if self._beats(single_mask):
                moves.append(single_mask)

        # Pairs (vectorized)
        pairs = self._find_pairs_vectorized(hand_cards)
        for pair in pairs:
            # Check game start requirement
            if is_game_start and not pair[three_of_diamonds]:
                continue  # Must include 3♦ in starting play

            if self._beats(pair):
                moves.append(pair)

        # Trips (vectorized)
        trips = self._find_trips_vectorized(hand_cards)
        for trip in trips:
            # Check game start requirement
            if is_game_start and not trip[three_of_diamonds]:
                continue  # Must include 3♦ in starting play

            if self._beats(trip):
                moves.append(trip)

        # 5-card hands (with early exit heuristics)
        if len(hand_cards) >= 5:
            # Early exit: If last play is 5-card, check if we can even beat it
            if self.last_play is not None and np.sum(self.last_play[0]) == 5:
                last_play_cards = np.where(self.last_play[0])[0]
                last_type, last_strength = self._identify_hand_type(last_play_cards)
                # Skip expensive computation if we clearly can't beat it
                if not self._can_potentially_beat_5_card(hand_cards, last_type, last_strength):
                    pass  # Skip 5-card generation entirely
                else:
                    five_card_hands = self._generate_5_card_hands_optimized(hand_cards)
                    # Filter for game start requirement
                    for hand in five_card_hands:
                        if is_game_start and not hand[three_of_diamonds]:
                            continue  # Must include 3♦ in starting play
                        moves.append(hand)
            else:
                # No last play or last play wasn't 5-card, generate all 5-card hands
                five_card_hands = self._generate_5_card_hands_optimized(hand_cards)
                # Filter for game start requirement
                for hand in five_card_hands:
                    if is_game_start and not hand[three_of_diamonds]:
                        continue  # Must include 3♦ in starting play
                    moves.append(hand)

        # Always allowed to pass (unless starting the very first move of the game)
        if self.last_play is not None:
            pass_mask = np.zeros(52, dtype=bool)
            moves.append(pass_mask)  # PASS as all-false mask

        return moves

    def _get_move_type(self, move_array: np.ndarray) -> str:
        """Get descriptive name for a move type."""
        move_indices = np.where(move_array)[0]
        move_len = len(move_indices)

        if move_len == 0:
            return "Pass"
        elif move_len == 1:
            return "Single"
        elif move_len == 2:
            return "Pair"
        elif move_len == 3:
            return "Trips"
        elif move_len == 5:
            hand_type, _ = self._identify_hand_type(move_indices)
            return hand_type.replace("_", " ").title()
        else:
            return f"{move_len}-card"

    def _beats(self, play: Union[List[int], np.ndarray]) -> bool:
        """Check if a candidate play beats last_play, or starts a trick if no last_play."""
        # Interpret PASS first
        is_pass = False
        if isinstance(play, np.ndarray):
            is_pass = not np.any(play)
        else:
            is_pass = play == []

        # Handle starting new trick vs pass move
        if self.last_play is None:
            # Starting new trick - pass is not allowed
            if is_pass:
                return False
            # Any valid hand can start a trick
            return True

        # There is a last play - pass is allowed
        if is_pass:
            return True

        last_play_cards = np.where(self.last_play[0])[0].tolist()
        # Normalize play to list of indices
        if isinstance(play, np.ndarray):
            play_indices = np.where(play)[0].tolist()
        else:
            play_indices = play
        if len(play_indices) != len(last_play_cards):
            return False

        # Get hand types and strengths - both inputs are now lists
        play_type, play_strength = self._identify_hand_type(play_indices)
        last_type, last_strength = self._identify_hand_type(last_play_cards)

        # Invalid hands can't beat anything
        if play_type == "invalid":
            return False

        # For 5-card hands, different types have different base strengths
        if len(play_indices) == 5:
            type_hierarchy = {
                "straight": 1,
                "flush": 2,
                "full_house": 3,
                "four_of_a_kind": 4,
                "straight_flush": 5,
            }

            play_base = type_hierarchy.get(play_type, 0)
            last_base = type_hierarchy.get(last_type, 0)

            if play_base > last_base:
                return True
            elif play_base < last_base:
                return False
            # Same type, compare by strength value

        # For same type or non-5-card hands, compare by strength
        return play_strength > last_strength

    def step(self, action: int) -> Tuple[Dict[str, Any], List[float], bool, Dict[str, Any]]:
        """Action = index into legal_moves list."""
        if self.done:
            raise ValueError("Game already over")

        player = self.current_player
        moves = self.legal_moves(player)
        move = moves[action]

        # Support numpy mask or list of indices
        if isinstance(move, np.ndarray):
            if np.any(move):
                # Remove played cards from hand
                self.hands[player, move] = False
                # Store last play as boolean array directly
                last_play_array = move.astype(bool)
                self.last_play = (last_play_array, player)
                self.passes_in_row = 0

                # Add to move history (keep last 5 moves) - only if enabled
                if self.track_move_history:
                    move_type = self._get_move_type(last_play_array)
                    self.move_history.append((last_play_array.copy(), player, move_type))
                    if len(self.move_history) > 5:
                        self.move_history.pop(0)
            else:
                # PASS
                self.passes_in_row += 1
                if self.passes_in_row == self.num_players - 1:
                    # trick reset
                    self.last_play = None
                    self.passes_in_row = 0

                # Add pass to move history - only if enabled
                if self.track_move_history:
                    pass_array = np.zeros(52, dtype=bool)
                    self.move_history.append((pass_array.copy(), player, "Pass"))
                    if len(self.move_history) > 5:
                        self.move_history.pop(0)
        elif move:  # list of indices non-empty
            # Remove played cards from hand (set to False)
            self.hands[player, move] = False
            # Store last play as boolean array
            last_play_array = np.zeros(52, dtype=bool)
            last_play_array[move] = True
            self.last_play = (last_play_array, player)
            self.passes_in_row = 0
        else:  # PASS with empty list
            self.passes_in_row += 1
            if self.passes_in_row == self.num_players - 1:
                # trick reset
                self.last_play = None
                self.passes_in_row = 0

        # Check win
        reward = [0] * self.num_players
        if np.sum(self.hands[player]) == 0:  # no cards left
            self.done = True
            # New reward structure
            for p in range(self.num_players):
                cards_left = np.sum(self.hands[p])  # count True values
                if p == player:
                    # Winner gets massive reward
                    reward[p] = 5
                else:
                    # Non-winners: reward based on cards remaining (nonlinear)
                    if cards_left >= 10:
                        reward[p] = cards_left * -3  # Hugely negative for 10+ cards
                    elif cards_left >= 5:
                        reward[p] = cards_left * -1.5  # Negative for 5-9 cards
                    elif cards_left >= 2:
                        reward[p] = cards_left * -1  # Small penalty for 1-4 cards
                    else:
                        reward[p] = 0

        # Rotate turn
        self.current_player = (self.current_player + 1) % self.num_players
        return self._get_obs(), reward, self.done, {}

    def _can_potentially_beat_5_card(self, hand: List[int], last_type: str, _last_strength: int) -> bool:
        """Quick check if hand could potentially beat a 5-card play."""
        # Quick heuristics to avoid expensive computation

        # Count cards by suit and rank using vectorized operations
        if len(hand) == 0:
            return False

        hand_arr = np.array(hand)
        suits_arr = hand_arr % 4
        ranks_arr = hand_arr // 4

        # Count occurrences using numpy
        suits = {}
        ranks = {}
        for suit in suits_arr:
            suits[suit] = suits.get(suit, 0) + 1
        for rank in ranks_arr:
            ranks[rank] = ranks.get(rank, 0) + 1

        # Check for potential flushes (5+ cards of same suit)
        has_flush_potential = any(count >= 5 for count in suits.values())

        # Check for potential pairs/trips for full house
        rank_counts = list(ranks.values())
        has_pair_potential = any(count >= 2 for count in rank_counts)
        has_trip_potential = any(count >= 3 for count in rank_counts)

        # Check for potential straights (5+ consecutive ranks)
        sorted_ranks = sorted(ranks.keys())
        has_straight_potential = len(sorted_ranks) >= 5 and (max(sorted_ranks) - min(sorted_ranks) <= 8)

        # Quick decisions based on last play type
        if last_type == "straight_flush":
            # Need straight flush or better - very rare, usually skip
            return has_flush_potential and has_straight_potential
        elif last_type == "four_of_a_kind":
            # Need four of a kind or straight flush
            return any(count >= 4 for count in rank_counts) or (has_flush_potential and has_straight_potential)
        elif last_type == "full_house":
            # Need full house or better
            return (
                (has_trip_potential and has_pair_potential)
                or any(count >= 4 for count in rank_counts)
                or (has_flush_potential and has_straight_potential)
            )
        elif last_type == "flush":
            # Need flush or better
            return (
                has_flush_potential
                or has_trip_potential
                or any(count >= 4 for count in rank_counts)
                or has_straight_potential
            )
        elif last_type == "straight":
            # Need straight or better
            return (
                has_straight_potential
                or has_flush_potential
                or has_trip_potential
                or any(count >= 4 for count in rank_counts)
            )

        return True  # Default to checking if unsure

    def _generate_5_card_hands_optimized(self, hand: List[int]) -> List[np.ndarray]:
        """Generate 5-card hands with smart pre-filtering to avoid expensive combinations."""
        moves = []

        # Quick exit if not enough cards
        if len(hand) < 5:
            return moves

        # Pre-analyze hand structure using vectorized operations
        hand_arr = np.array(hand)
        suits_arr = hand_arr % 4
        ranks_arr = hand_arr // 4

        # Vectorized counting
        unique_suits, suit_counts = np.unique(suits_arr, return_counts=True)
        unique_ranks, rank_counts = np.unique(ranks_arr, return_counts=True)

        # Create efficient lookups
        suit_dict = dict(zip(unique_suits, suit_counts))
        rank_dict = dict(zip(unique_ranks, rank_counts))

        # 1. FLUSHES: Only check suits with 5+ cards
        flush_suits = [suit for suit, count in suit_dict.items() if count >= 5]
        for suit in flush_suits:
            suit_cards = hand_arr[suits_arr == suit].tolist()
            self._generate_flush_combinations(suit_cards, moves)

        # 2. FOUR OF A KIND: Only check ranks with 4+ cards
        quad_ranks = [rank for rank, count in rank_dict.items() if count >= 4]
        for rank in quad_ranks:
            rank_cards = hand_arr[ranks_arr == rank].tolist()
            other_cards = hand_arr[ranks_arr != rank].tolist()
            for quad_combo in combinations(rank_cards, 4):
                for kicker in other_cards:
                    combo_mask = np.zeros(52, dtype=bool)
                    combo_mask[list(quad_combo)] = True
                    combo_mask[kicker] = True
                    if self._beats(combo_mask):
                        moves.append(combo_mask)

        # 3. FULL HOUSE: Only check ranks with 3+ and 2+ cards
        trip_ranks = [rank for rank, count in rank_dict.items() if count >= 3]
        pair_ranks = [rank for rank, count in rank_dict.items() if count >= 2]

        for trip_rank in trip_ranks:
            trip_cards = hand_arr[ranks_arr == trip_rank].tolist()
            for trip_combo in combinations(trip_cards, 3):
                for pair_rank in pair_ranks:
                    if pair_rank != trip_rank:
                        pair_cards = hand_arr[ranks_arr == pair_rank].tolist()
                        for pair_combo in combinations(pair_cards, 2):
                            combo_mask = np.zeros(52, dtype=bool)
                            combo_mask[list(trip_combo)] = True
                            combo_mask[list(pair_combo)] = True
                            if self._beats(combo_mask):
                                moves.append(combo_mask)

        # 4. STRAIGHTS: Smart straight detection
        if self._has_straight_potential(unique_ranks):
            self._generate_straight_combinations(hand, moves)

        return moves

    def _generate_flush_combinations(self, suit_cards: List[int], moves: List[np.ndarray]) -> None:
        """Generate valid flush combinations from cards of same suit."""
        for combo in combinations(suit_cards, 5):
            combo_mask = np.zeros(52, dtype=bool)
            combo_mask[list(combo)] = True
            if self._beats(combo_mask):
                moves.append(combo_mask)

    def _has_straight_potential(self, unique_ranks: List[int]) -> bool:
        """Quick check if hand has potential for straights."""
        if len(unique_ranks) < 5:
            return False

        # Convert to sorted array for efficient checking
        ranks = np.sort(unique_ranks)

        # Check for 5+ consecutive ranks
        for i in range(len(ranks) - 4):
            if ranks[i + 4] - ranks[i] == 4:
                return True

        # Check for A-2-3-4-5 straight (ranks 0,1,2,11,12)
        ace_low_straight = np.array([0, 1, 2, 11, 12])
        if np.all(np.isin(ace_low_straight, ranks)):
            return True

        return False

    def _generate_straight_combinations(self, hand: List[int], moves: List[np.ndarray]) -> None:
        """Generate straight combinations efficiently."""
        # Pre-filter: only generate combinations that could form straights
        hand_arr = np.array(hand)
        ranks_arr = hand_arr // 4

        # Group cards by rank for efficient straight building
        rank_groups = {}
        for i, rank in enumerate(ranks_arr):
            if rank not in rank_groups:
                rank_groups[rank] = []
            rank_groups[rank].append(hand[i])

        # Generate straights more efficiently by building them rank by rank
        sorted_ranks = sorted(rank_groups.keys())

        # Check regular straights
        for start_idx in range(len(sorted_ranks) - 4):
            if sorted_ranks[start_idx + 4] - sorted_ranks[start_idx] == 4:
                straight_ranks = sorted_ranks[start_idx : start_idx + 5]
                self._build_straights_from_ranks(straight_ranks, rank_groups, moves)

        # Check A-2-3-4-5 straight
        ace_low_ranks = [0, 1, 2, 11, 12]
        if all(rank in rank_groups for rank in ace_low_ranks):
            self._build_straights_from_ranks(ace_low_ranks, rank_groups, moves)

    def _build_straights_from_ranks(
        self,
        straight_ranks: List[int],
        rank_groups: Dict[int, List[int]],
        moves: List[np.ndarray],
    ) -> None:
        """Build all possible straights from given ranks."""
        # Get one card from each rank to form straights
        rank_choices = [rank_groups[rank] for rank in straight_ranks]

        # Generate all combinations of one card per rank
        for combo in self._cartesian_product(rank_choices):
            combo_mask = np.zeros(52, dtype=bool)
            combo_mask[list(combo)] = True
            if self._beats(combo_mask):
                moves.append(combo_mask)

    def _cartesian_product(self, lists: List[List[int]]) -> Generator[List[int], None, None]:
        """Efficient cartesian product for straight generation."""
        if not lists:
            yield []
            return

        for item in lists[0]:
            for rest in self._cartesian_product(lists[1:]):
                yield [item] + rest

    def _find_pairs_vectorized(self, hand: List[int]) -> List[np.ndarray]:
        """Find all pairs using fully vectorized numpy operations."""
        if len(hand) < 2:
            return []

        hand_arr = np.array(hand)
        ranks = hand_arr // 4  # Vectorized rank calculation

        # Find all unique ranks and their counts
        unique_ranks, counts = np.unique(ranks, return_counts=True)

        pairs = []
        # For each rank with 2+ cards, generate pairs
        for rank, count in zip(unique_ranks, counts):
            if count >= 2:
                # Get all cards of this rank
                rank_cards = hand_arr[ranks == rank]

                # Generate all pairs from this rank
                def add_pair(a: int, b: int) -> None:
                    mask = np.zeros(52, dtype=bool)
                    mask[[a, b]] = True
                    pairs.append(mask)

                add_pair(rank_cards[0], rank_cards[1])
                if count >= 3:
                    add_pair(rank_cards[0], rank_cards[2])
                    add_pair(rank_cards[1], rank_cards[2])
                if count >= 4:
                    add_pair(rank_cards[2], rank_cards[3])

        return pairs

    def _find_trips_vectorized(self, hand: List[int]) -> List[np.ndarray]:
        """Find all trips using fully vectorized numpy operations."""
        if len(hand) < 3:
            return []

        hand_arr = np.array(hand)
        ranks = hand_arr // 4  # Vectorized rank calculation

        # Find all unique ranks and their counts
        unique_ranks, counts = np.unique(ranks, return_counts=True)

        trips = []
        # For each rank with 3+ cards, generate trips
        for rank, count in zip(unique_ranks, counts):
            if count >= 3:
                # Get all cards of this rank
                rank_cards = hand_arr[ranks == rank]

                def add_trip(a: int, b: int, c: int) -> None:
                    mask = np.zeros(52, dtype=bool)
                    mask[[a, b, c]] = True
                    trips.append(mask)

                add_trip(rank_cards[0], rank_cards[1], rank_cards[2])
                if count >= 4:
                    add_trip(rank_cards[0], rank_cards[1], rank_cards[3])

        return trips

    def _group_cards_by_rank_vectorized(self, hand: List[int]) -> Dict[int, List[int]]:
        """Group cards by rank using numpy for efficient processing."""
        if not hand:
            return {}

        hand_arr = np.array(hand)
        ranks = hand_arr // 4

        rank_groups = {}
        for i, rank in enumerate(ranks):
            if rank not in rank_groups:
                rank_groups[rank] = []
            rank_groups[rank].append(hand[i])

        return rank_groups

    def _group_cards_by_suit_vectorized(self, hand: List[int]) -> Dict[int, List[int]]:
        """Group cards by suit using numpy for efficient processing."""
        if not hand:
            return {}

        hand_arr = np.array(hand)
        suits = hand_arr % 4

        suit_groups = {}
        for i, suit in enumerate(suits):
            if suit not in suit_groups:
                suit_groups[suit] = []
            suit_groups[suit].append(hand[i])

        return suit_groups

    def _get_obs(self) -> np.ndarray:
        """Observation: current hand + full previous play as binary vectors."""
        player = self.current_player

        # Current hand as binary vector (already in correct format)
        hand_obs = self.hands[player].astype(np.int8)

        # Previous play as binary vector
        last_play_obs = np.zeros(52, dtype=np.int8)
        last_play_exists = 0
        if self.last_play is not None:
            last_play_obs = self.last_play[0].astype(np.int8)
            last_play_exists = 1

        return {
            "hand": hand_obs,
            "last_play": last_play_obs,
            "last_play_exists": last_play_exists,
            "legal_moves": self.legal_moves(player),
        }
