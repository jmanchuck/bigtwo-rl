# Training + Evaluation Runbook

This is the minimal, current-state guide for reproducing training and evaluation.

## Current Stack
- PPO: `MultiPlayerPPO` (`sb3_contrib.MaskablePPO` based)
- Env: `BigTwoWrapper` with strict illegal-action penalty and action masking
- Observation: `enhanced` only, `412` dims (reference-inspired)
- Opponent setup for KPI: 1 PPO agent vs 3 baseline agents

## Observation Schema (412)
Index layout:
- `0:286` current-hand slot features (`13 x 22`)
- `286:367` opponent context (`3 x 27`)
- `367:383` global high-card record (`16` for Q/K/A/2)
- `383:412` previous-hand/control context (`29`)

Key notes:
- Ruleset adaptation: no two-pair, triples supported.
- First-play/control semantics are aligned across masking, observation, and evaluation.

## Recommended Training Config
Use this as baseline for stable long runs:
- `n_envs=8`
- `learning_rate=2.5e-4`
- `gamma=0.995`
- `n_steps=64`
- `batch_size=256`
- `n_epochs=4`
- `ent_coef=0.005`
- `league_opponent_prob=0.0`

## Repro: Checkpoint-Gated Training
Use gating so we keep best checkpoint and avoid effective regression.

```bash
MPLCONFIGDIR=/tmp/.mpl ./.venv/bin/python examples/train_checkpoint_gated.py \
  --run-name run_name \
  --max-steps 200000 \
  --chunk-steps 50000 \
  --eval-games 200 \
  --behavior-games 80 \
  --seed 11
```

Artifacts:
- `models/<run_name>/checkpoint_<steps>.zip`
- `models/<run_name>/best_by_random.zip`
- `models/<run_name>/gated_metrics.json`
- `models/<run_name>/gated_summary.json`

## Repro: Direct Stochastic Evaluation
```bash
# vs random
MPLCONFIGDIR=/tmp/.mpl ./.venv/bin/python examples/benchmark_vs_random.py \
  --model models/<run_name>/checkpoint_<steps>.zip \
  --opponent random \
  --stochastic \
  --games 300 \
  --seed 11

# vs greedy
MPLCONFIGDIR=/tmp/.mpl ./.venv/bin/python examples/benchmark_vs_random.py \
  --model models/<run_name>/checkpoint_<steps>.zip \
  --opponent greedy \
  --stochastic \
  --games 300 \
  --seed 12
```

## KPI Policy
Primary KPI:
- stochastic win rate vs random agents

Secondary KPI:
- stochastic win rate vs greedy agents
- behavior metrics from `train_checkpoint_gated.py`:
  - `noncontrol_pass_when_nonpass`
  - `control_pair_rate`
  - `control_avg_cards_played`

## Known Stability Note
Long runs can hit masked-categorical numerical edge cases.
Current mitigation in `MultiPlayerPPO`:
- mask sanitization in rollout/buffer paths
- torch distribution arg validation disabled for training stability

Keep checkpoint gating enabled for all long runs.
