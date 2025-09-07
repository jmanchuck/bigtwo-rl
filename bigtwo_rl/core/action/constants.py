"""Action space constants for Big Two RL environment."""

from math import comb

# ============================================================
# Action space layout
# ============================================================

# Action indices in the 1365-dimensional action space:
# 0: Pass
# 1-13: Singles (13 actions)
# 14-46: Pairs (33 actions)
# 47-77: Triples (31 actions)
# 78-1364: Five-card combos (1287 actions = C(13,5))

OFF_PASS = 0
OFF_1 = 1
OFF_2 = OFF_1 + 13  # = 14
OFF_3 = OFF_2 + 33  # = 47
OFF_5 = OFF_3 + 31  # = 78
N_ACTIONS = OFF_5 + comb(13, 5)  # = 78 + 1287 = 1365

assert N_ACTIONS == 1365, f"Expected 1365 actions, got {N_ACTIONS}"
