# RL Current State (Feb 2026)

This file tracks only the current state that matters for ongoing work.

## What Is In Place
- Action masking is active in rollout and PPO optimization.
- Illegal actions are penalized (no silent action rewrite).
- Terminal rewards are back-assigned to the last 4 player transitions.
- Observation path is unified: `enhanced` only (`412` dims, reference-inspired).
- First-play/control semantics are consistent across env, observation, masking, eval.
- Checkpoint-gated training flow exists (`examples/train_checkpoint_gated.py`).

## Baseline Opponents
- `GreedyAgent` was fixed to prefer lowest non-pass legal move.
- Earlier greedy metrics produced before this fix are not reliable.

## Current Best Recent Result
From the ref-observation long resume run:
- checkpoint: `models/refobs412_curve_400k_feb20/checkpoint_300000_resume.zip`
- stochastic vs random: `0.43` (400 games, seed 1001)
- stochastic vs greedy: `0.065` (400 games, seed 1002)

Interpretation:
- Random-agent performance is now clearly above 25% baseline.
- Greedy-agent performance remains weak and is the major open gap.

## Open Gaps
1. Robustness vs greedy opponents.
2. Reduced variance across seeds/checkpoints.
3. More stable long-horizon training without numerical edge-case crashes.

## Current Operating Strategy
1. Train with checkpoint gating.
2. Promote best checkpoint by stochastic random win rate.
3. Revalidate promoted checkpoints with larger game counts and multiple seeds.
4. Track behavior metrics to ensure strategy quality, not just win-rate noise.

## Required Verification
- `./.venv/bin/pytest -q`
- Run gated training smoke before long runs.
- Run stochastic benchmark against both random and greedy before promoting.
