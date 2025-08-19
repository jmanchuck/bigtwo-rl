import random
import numpy as np
from itertools import combinations
from functools import lru_cache


class ToyBigTwoFullRules:
    # Precomputed lookup tables for card properties (massive speedup)
    _RANK_TABLE = [card // 4 for card in range(52)]  # 0 = 3, 12 = 2
    _SUIT_TABLE = [card % 4 for card in range(52)]   # 0 = diamonds, 1 = clubs, 2 = hearts, 3 = spades
    
    # Precomputed Big Two card values (2 is highest = 12, A = 11, etc.)
    _VALUE_TABLE = []
    _RANK_TO_VALUE = []  # Direct rank-to-value lookup
    
    for card in range(52):
        rank = card // 4
        if rank == 0:    # 3
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
        elif rank == 10: # K
            _VALUE_TABLE.append(10)
        elif rank == 11: # A
            _VALUE_TABLE.append(11)
        elif rank == 12: # 2
            _VALUE_TABLE.append(12)
    
    # Fill rank-to-value lookup (ranks 0-12 -> values 0-12)
    for rank in range(13):
        _RANK_TO_VALUE.append(rank)  # Since our rank mapping already matches Big Two values
    
    def __init__(self, num_players=4):
        self.num_players = num_players
        self.reset()

    def reset(self, seed=None):
        if seed:
            random.seed(seed)
        # Deal cards
        deck = list(range(52))  # 0..51 = 13 ranks × 4 suits
        random.shuffle(deck)
        self.hands = [deck[i :: self.num_players] for i in range(self.num_players)]
        for h in self.hands:
            h.sort()
        # Game state
        self.current_player = 0
        self.last_play = None  # (cards, player)
        self.passes_in_row = 0
        self.done = False
        return self._get_obs()

    def _rank(self, card):
        return self._RANK_TABLE[card]
    
    def _suit(self, card):
        return self._SUIT_TABLE[card]
    
    def _card_value(self, card):
        """Get Big Two card value where 2 is highest (12), A is second highest (11), etc."""
        return self._VALUE_TABLE[card]
        
    @lru_cache(maxsize=50000)
    def _identify_hand_type_cached(self, cards_tuple):
        """Cached hand type identification using LRU cache."""
        return self._identify_hand_type_uncached(list(cards_tuple))
    
    def _identify_hand_type(self, cards):
        """Identify the type of hand and return (hand_type, strength_value)."""
        # Use LRU cache for frequently computed hand types
        cards_tuple = tuple(sorted(cards))
        return self._identify_hand_type_cached(cards_tuple)
    
    def _identify_hand_type_uncached(self, cards):
        """Actual hand type identification without caching."""
        if len(cards) == 1:
            return "single", self._card_value(cards[0]) * 4 + self._suit(cards[0])
        elif len(cards) == 2:
            if self._rank(cards[0]) == self._rank(cards[1]):
                return "pair", self._card_value(cards[0]) * 4 + max(self._suit(cards[0]), self._suit(cards[1]))
            else:
                return "invalid", 0
        elif len(cards) == 3:
            ranks = [self._rank(c) for c in cards]
            if len(set(ranks)) == 1:  # All same rank
                return "trips", self._card_value(cards[0]) * 4 + max(self._suit(c) for c in cards)
            else:
                return "invalid", 0
        elif len(cards) == 5:
            return self._identify_five_card_hand(cards)
        else:
            return "invalid", 0
    
    def _identify_five_card_hand(self, cards):
        """Identify 5-card hand types and return strength."""
        ranks = sorted([self._rank(c) for c in cards])
        suits = [self._suit(c) for c in cards]
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        
        is_flush = len(set(suits)) == 1
        is_straight = self._is_straight(ranks)
        
        # Check for straight flush
        if is_straight and is_flush:
            highest_card = max(cards, key=lambda c: self._card_value(c) * 4 + self._suit(c))
            return "straight_flush", 50000 + self._card_value(highest_card) * 4 + self._suit(highest_card)
        
        # Check for four of a kind
        if 4 in rank_counts.values():
            four_rank = [r for r, count in rank_counts.items() if count == 4][0]
            return "four_of_a_kind", 40000 + self._card_value_from_rank(four_rank) * 4
        
        # Check for full house
        if 3 in rank_counts.values() and 2 in rank_counts.values():
            three_rank = [r for r, count in rank_counts.items() if count == 3][0]
            return "full_house", 30000 + self._card_value_from_rank(three_rank) * 4
        
        # Check for flush
        if is_flush:
            # Compare by highest card, then suit
            highest_card = max(cards, key=lambda c: self._card_value(c) * 4 + self._suit(c))
            return "flush", 20000 + self._card_value(highest_card) * 4 + self._suit(highest_card)
        
        # Check for straight
        if is_straight:
            highest_card = max(cards, key=lambda c: self._card_value(c))
            return "straight", 10000 + self._card_value(highest_card) * 4 + self._suit(highest_card)
        
        return "invalid", 0
    
    def _card_value_from_rank(self, rank):
        """Convert rank (0-12) to Big Two value (0-12)."""
        return self._RANK_TO_VALUE[rank]
    
    def _is_straight(self, ranks):
        """Check if ranks form a valid Big Two straight."""
        # Convert to Big Two values for straight checking
        values = [self._card_value_from_rank(r) for r in ranks]
        values.sort()
        
        # Check for regular straight
        for i in range(1, len(values)):
            if values[i] != values[i-1] + 1:
                break
        else:
            return True
        
        # Check for A-2-3-4-5 straight (values would be [0,1,2,11,12])
        if values == [0, 1, 2, 11, 12]:  # 3-4-5-A-2
            return True
            
        return False

    def legal_moves(self, player):
        """Generate all legal singles, pairs, trips, and 5-card hands."""
        hand = self.hands[player]
        moves = []

        # Singles
        for c in hand:
            if self._beats([c]):
                moves.append([c])

        # Pairs (vectorized)
        pairs = self._find_pairs_vectorized(hand)
        for pair in pairs:
            if self._beats(pair):
                moves.append(pair)

        # Trips (vectorized)
        trips = self._find_trips_vectorized(hand)
        for trip in trips:
            if self._beats(trip):
                moves.append(trip)

        # 5-card hands (with early exit heuristics)
        if len(hand) >= 5:
            # Early exit: If last play is 5-card, check if we can even beat it
            if self.last_play is not None and len(self.last_play[0]) == 5:
                last_type, last_strength = self._identify_hand_type(self.last_play[0])
                # Skip expensive computation if we clearly can't beat it
                if not self._can_potentially_beat_5_card(hand, last_type, last_strength):
                    pass  # Skip 5-card generation entirely
                else:
                    moves.extend(self._generate_5_card_hands_optimized(hand))
            else:
                # No last play or last play wasn't 5-card, generate all 5-card hands
                moves.extend(self._generate_5_card_hands_optimized(hand))

        # Always allowed to pass (unless starting new trick)
        if self.last_play is not None:
            moves.append([])  # [] = PASS

        return moves

    def _beats(self, play):
        """Check if a candidate play beats last_play, or starts a trick if no last_play."""
        if self.last_play is None:
            return True
        if play == []:  # pass is always legal (but handled separately)
            return True
        if len(play) != len(self.last_play[0]):
            return False
        
        # Get hand types and strengths
        play_type, play_strength = self._identify_hand_type(play)
        last_type, last_strength = self._identify_hand_type(self.last_play[0])
        
        # Invalid hands can't beat anything
        if play_type == "invalid":
            return False
        
        # For 5-card hands, different types have different base strengths
        if len(play) == 5:
            type_hierarchy = {
                "straight": 1,
                "flush": 2,
                "full_house": 3,
                "four_of_a_kind": 4,
                "straight_flush": 5
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

    def step(self, action):
        """Action = index into legal_moves list."""
        if self.done:
            raise ValueError("Game already over")

        player = self.current_player
        moves = self.legal_moves(player)
        move = moves[action]

        if move:  # played cards
            for c in move:
                self.hands[player].remove(c)
            self.last_play = (move, player)
            self.passes_in_row = 0
        else:  # PASS
            self.passes_in_row += 1
            if self.passes_in_row == self.num_players - 1:
                # trick reset
                self.last_play = None
                self.passes_in_row = 0

        # Check win
        reward = [0] * self.num_players
        if len(self.hands[player]) == 0:
            self.done = True
            # New reward structure
            for p in range(self.num_players):
                cards_left = len(self.hands[p])
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

    def _can_potentially_beat_5_card(self, hand, last_type, _last_strength):
        """Quick check if hand could potentially beat a 5-card play."""
        # Quick heuristics to avoid expensive computation
        
        # Count cards by suit and rank using vectorized operations
        if not hand:
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
            return (has_trip_potential and has_pair_potential) or any(count >= 4 for count in rank_counts) or (has_flush_potential and has_straight_potential)
        elif last_type == "flush":
            # Need flush or better
            return has_flush_potential or has_trip_potential or any(count >= 4 for count in rank_counts) or has_straight_potential
        elif last_type == "straight":
            # Need straight or better
            return has_straight_potential or has_flush_potential or has_trip_potential or any(count >= 4 for count in rank_counts)
        
        return True  # Default to checking if unsure
    
    def _generate_5_card_hands_optimized(self, hand):
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
                    combo_list = list(quad_combo) + [kicker]
                    if self._beats(combo_list):
                        moves.append(combo_list)
        
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
                            combo_list = list(trip_combo) + list(pair_combo)
                            if self._beats(combo_list):
                                moves.append(combo_list)
        
        # 4. STRAIGHTS: Smart straight detection
        if self._has_straight_potential(unique_ranks):
            self._generate_straight_combinations(hand, moves)
        
        return moves
    
    def _generate_flush_combinations(self, suit_cards, moves):
        """Generate valid flush combinations from cards of same suit."""
        for combo in combinations(suit_cards, 5):
            combo_list = list(combo)
            if self._beats(combo_list):
                moves.append(combo_list)
    
    def _has_straight_potential(self, unique_ranks):
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
    
    def _generate_straight_combinations(self, hand, moves):
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
                straight_ranks = sorted_ranks[start_idx:start_idx + 5]
                self._build_straights_from_ranks(straight_ranks, rank_groups, moves)
        
        # Check A-2-3-4-5 straight
        ace_low_ranks = [0, 1, 2, 11, 12]
        if all(rank in rank_groups for rank in ace_low_ranks):
            self._build_straights_from_ranks(ace_low_ranks, rank_groups, moves)
    
    def _build_straights_from_ranks(self, straight_ranks, rank_groups, moves):
        """Build all possible straights from given ranks."""
        # Get one card from each rank to form straights
        rank_choices = [rank_groups[rank] for rank in straight_ranks]
        
        # Generate all combinations of one card per rank
        for combo in self._cartesian_product(rank_choices):
            combo_list = list(combo)
            if self._beats(combo_list):
                moves.append(combo_list)
    
    def _cartesian_product(self, lists):
        """Efficient cartesian product for straight generation."""
        if not lists:
            yield []
            return
        
        for item in lists[0]:
            for rest in self._cartesian_product(lists[1:]):
                yield [item] + rest
    
    def _find_pairs_vectorized(self, hand):
        """Find all pairs using numpy operations for speed."""
        if len(hand) < 2:
            return []
        
        hand_arr = np.array(hand)
        ranks = hand_arr // 4  # Vectorized rank calculation
        
        pairs = []
        # Find consecutive equal ranks (since hand is sorted)
        for i in range(len(ranks) - 1):
            if ranks[i] == ranks[i + 1]:
                pairs.append([hand[i], hand[i + 1]])
        
        return pairs
    
    def _find_trips_vectorized(self, hand):
        """Find all trips using numpy operations for speed."""
        if len(hand) < 3:
            return []
        
        hand_arr = np.array(hand)
        ranks = hand_arr // 4  # Vectorized rank calculation
        
        trips = []
        # Find consecutive equal ranks (since hand is sorted)
        for i in range(len(ranks) - 2):
            if ranks[i] == ranks[i + 1] == ranks[i + 2]:
                trips.append([hand[i], hand[i + 1], hand[i + 2]])
        
        return trips
    
    def _group_cards_by_rank_vectorized(self, hand):
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
    
    def _group_cards_by_suit_vectorized(self, hand):
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

    def _get_obs(self):
        """Observation: current hand + full previous play as binary vectors."""
        player = self.current_player
        
        # Current hand as binary vector
        hand_obs = np.zeros(52, dtype=np.int8)
        hand_obs[self.hands[player]] = 1
        
        # Previous play as binary vector
        last_play_obs = np.zeros(52, dtype=np.int8)
        last_play_exists = 0
        if self.last_play is not None:
            last_play_obs[self.last_play[0]] = 1
            last_play_exists = 1
            
        return {
            "hand": hand_obs,
            "last_play": last_play_obs,
            "last_play_exists": last_play_exists,
            "legal_moves": self.legal_moves(player),
        }
