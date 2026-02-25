"""Train with stochastic checkpoint gating to prevent effective KPI regression.

This script trains in chunks, evaluates each checkpoint against random/greedy,
and promotes checkpoints by configurable KPI(s).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from bigtwo_rl.agents.ppo_agent import PPOAgent
from bigtwo_rl.agents.random_agent import RandomAgent
from bigtwo_rl.core.action import OFF_PASS, ActionMaskBuilder, BitsetFiveCardEngine, action_to_tuple
from bigtwo_rl.core.bigtwo import ToyBigTwoFullRules
from bigtwo_rl.core.observation import EnhancedObservationBuilder
from bigtwo_rl.training.multi_player_ppo import MultiPlayerPPO
from bigtwo_rl.training.self_play_callback import SimpleSelfPlayCallback
from bigtwo_rl.training.trainer import Trainer
from bigtwo_rl.training.rewards import (
    DefaultReward,
    ProgressiveReward,
    RankingReward,
    ScoreMarginReward,
    SparseReward,
)

try:
    from examples.benchmark_vs_random import evaluate
except ModuleNotFoundError:
    from benchmark_vs_random import evaluate


def build_model(trainer: Trainer, log_dir: Path) -> MultiPlayerPPO:
    env = DummyVecEnv([trainer._make_env_fn() for _ in range(max(1, trainer.n_envs))])
    policy_kwargs = {"activation_fn": __import__("torch").nn.ReLU, "net_arch": trainer.policy_net_arch}

    lr = trainer.learning_rate
    if trainer.anneal_lr:
        lr_init = trainer.learning_rate
        lr = lambda progress: max(1e-6, lr_init * progress)

    clip = trainer.clip_range
    if trainer.anneal_clip:
        clip_init = trainer.clip_range
        clip = lambda progress: clip_init * progress

    return MultiPlayerPPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        gamma=trainer.gamma,
        n_steps=trainer.n_steps,
        batch_size=trainer.batch_size,
        n_epochs=trainer.n_epochs,
        clip_range=clip,
        ent_coef=trainer.ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device=trainer.device,
        tensorboard_log=str(log_dir),
        league_opponent_prob=trainer.league_opponent_prob,
        snapshot_interval_rollouts=trainer.snapshot_interval_rollouts,
        snapshot_max_policies=trainer.snapshot_max_policies,
    )


def behavior_probe(model_path: str, games: int, seed: int) -> dict[str, float]:
    ppo = PPOAgent(model_path, deterministic=False)
    opps = [RandomAgent("R1"), RandomAgent("R2"), RandomAgent("R3")]
    rng = np.random.default_rng(seed)
    masker = ActionMaskBuilder(BitsetFiveCardEngine())
    obs_builder = EnhancedObservationBuilder()

    turns = 0
    pass_count = 0
    noncontrol_nonpass_available = 0
    noncontrol_pass_when_nonpass = 0

    control_turns = 0
    control_passes = 0
    control_nonpass = 0
    control_cards_played_total = 0
    control_pairs = 0
    control_triples = 0
    control_fives = 0

    for _ in range(games):
        game = ToyBigTwoFullRules(num_players=4, track_move_history=True)
        game.reset(seed=int(rng.integers(1_000_000_000)))

        for _ in range(500):
            if game.done:
                break
            player = game.current_player
            hand = game.get_player_hand(player)
            counts = game.get_player_card_counts()
            last_played = game.get_last_played_cards_encoded()
            is_first = (game.last_play is None) and all(c == 13 for c in counts)
            has_control = (game.last_play is None) and (not is_first)

            obs = obs_builder.build_observation(
                hand=hand,
                current_player=player,
                player_card_counts=counts,
                last_played_cards=last_played,
                passes=game.passes_in_row,
                is_first_play=is_first,
                move_history=getattr(game, "move_history", None),
                has_control=has_control,
                can_pass=(not is_first),
            )
            valid_ids = masker.full_mask_indices(
                hand,
                last_played,
                pass_allowed=not is_first,
                is_first_play=is_first,
                has_control=has_control,
            )
            mask = np.zeros(1365, dtype=bool)
            for idx in valid_ids:
                if 0 <= idx < 1365:
                    mask[idx] = True

            if not mask.any():
                action = OFF_PASS
            elif player == 0:
                action = ppo.get_action(obs, mask)
            else:
                action = opps[player - 1].get_action(obs, mask)

            if player == 0:
                turns += 1
                if action == OFF_PASS:
                    pass_count += 1

                if has_control:
                    control_turns += 1
                    if action == OFF_PASS:
                        control_passes += 1
                    else:
                        control_nonpass += 1
                        slots = list(action_to_tuple(action))
                        move_size = 0
                        for s in slots:
                            if 0 <= s < len(hand.card) and not hand.played[s]:
                                move_size += 1
                        control_cards_played_total += move_size
                        if move_size == 2:
                            control_pairs += 1
                        elif move_size == 3:
                            control_triples += 1
                        elif move_size == 5:
                            control_fives += 1
                else:
                    if np.any(mask[1:]):
                        noncontrol_nonpass_available += 1
                        if action == OFF_PASS:
                            noncontrol_pass_when_nonpass += 1

            slots = [] if action == OFF_PASS else list(action_to_tuple(action))
            if not game.play_hand_move(player, slots):
                game.play_hand_move(player, [])

    return {
        "pass_rate": pass_count / max(1, turns),
        "noncontrol_pass_when_nonpass": noncontrol_pass_when_nonpass / max(1, noncontrol_nonpass_available),
        "control_pass_rate": control_passes / max(1, control_turns),
        "control_pair_rate": control_pairs / max(1, control_nonpass),
        "control_triple_rate": control_triples / max(1, control_nonpass),
        "control_five_rate": control_fives / max(1, control_nonpass),
        "control_avg_cards_played": control_cards_played_total / max(1, control_nonpass),
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _league_prob_for_step(
    trained_steps: int,
    start_prob: float,
    end_prob: float,
    ramp_steps: int,
) -> float:
    if ramp_steps <= 0:
        return _clamp01(end_prob)
    frac = min(1.0, max(0.0, trained_steps / ramp_steps))
    return _clamp01(start_prob + (end_prob - start_prob) * frac)


def _kpi_score(wr_random: float, wr_greedy: float, mode: str, greedy_weight: float) -> float:
    if mode == "random":
        return wr_random
    if mode == "greedy":
        return wr_greedy
    g_weight = _clamp01(greedy_weight)
    return (1.0 - g_weight) * wr_random + g_weight * wr_greedy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="gated_curve")
    parser.add_argument("--max-steps", type=int, default=200_000)
    parser.add_argument("--chunk-steps", type=int, default=50_000)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--behavior-games", type=int, default=80)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--games-per-episode", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument(
        "--league-opponent-prob",
        type=float,
        default=None,
        help="Deprecated constant value; overrides start/end/ramp when set.",
    )
    parser.add_argument("--league-opponent-prob-start", type=float, default=0.0)
    parser.add_argument("--league-opponent-prob-end", type=float, default=0.12)
    parser.add_argument("--league-opponent-prob-ramp-steps", type=int, default=1_000_000)
    parser.add_argument("--primary-kpi", choices=["greedy", "random", "blended"], default="greedy")
    parser.add_argument(
        "--blended-kpi-greedy-weight",
        type=float,
        default=0.7,
        help="Only used when --primary-kpi=blended.",
    )
    parser.add_argument(
        "--reward",
        choices=["default", "sparse", "progressive", "ranking", "score_margin"],
        default="default",
    )
    parser.add_argument("--no-anneal-lr", action="store_true")
    parser.add_argument("--no-anneal-clip", action="store_true")
    args = parser.parse_args()
    if args.league_opponent_prob is not None:
        league_prob_start = _clamp01(args.league_opponent_prob)
        league_prob_end = league_prob_start
        league_prob_ramp_steps = 0
    else:
        league_prob_start = _clamp01(args.league_opponent_prob_start)
        league_prob_end = _clamp01(args.league_opponent_prob_end)
        league_prob_ramp_steps = max(0, int(args.league_opponent_prob_ramp_steps))
    blended_kpi_greedy_weight = _clamp01(args.blended_kpi_greedy_weight)

    model_dir = Path("models") / args.run_name
    log_dir = Path("logs") / args.run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    reward_map = {
        "default": DefaultReward,
        "sparse": SparseReward,
        "progressive": ProgressiveReward,
        "ranking": RankingReward,
        "score_margin": ScoreMarginReward,
    }
    reward_fn = reward_map[args.reward]()

    trainer = Trainer(
        reward_function=reward_fn,
        n_envs=args.n_envs,
        games_per_episode=args.games_per_episode,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        league_opponent_prob=league_prob_start,
        anneal_lr=not args.no_anneal_lr,
        anneal_clip=not args.no_anneal_clip,
    )

    model = build_model(trainer, log_dir)
    callback = CallbackList([SimpleSelfPlayCallback(verbose=0)])

    trained_steps = 0
    best_random = -1.0
    best_random_checkpoint = None
    best_greedy = -1.0
    best_greedy_checkpoint = None
    best_blended = -1.0
    best_blended_checkpoint = None
    best_primary = -1.0
    best_primary_checkpoint = None
    records: list[dict] = []

    while trained_steps < args.max_steps:
        model.league_opponent_prob = _league_prob_for_step(
            trained_steps=trained_steps,
            start_prob=league_prob_start,
            end_prob=league_prob_end,
            ramp_steps=league_prob_ramp_steps,
        )

        step = min(args.chunk_steps, args.max_steps - trained_steps)
        # Keep each chunk as an independent learn horizon for schedule stability.
        # Reusing reset_num_timesteps=False across repeated short learns can push
        # progress-based schedules into unstable regions.
        model.learn(total_timesteps=step, callback=callback, progress_bar=True, reset_num_timesteps=True)
        trained_steps += step

        ckpt_base = model_dir / f"checkpoint_{trained_steps}"
        model.save(ckpt_base)
        ckpt = str(ckpt_base) + ".zip"

        wr_random = evaluate(
            ckpt,
            games=args.eval_games,
            strategy="argmax",
            seed=args.seed + trained_steps // 1000,
            opponent="random",
            stochastic=True,
        )
        wr_greedy = evaluate(
            ckpt,
            games=args.eval_games,
            strategy="argmax",
            seed=args.seed + 10_000 + trained_steps // 1000,
            opponent="greedy",
            stochastic=True,
        )
        score_blended = _kpi_score(wr_random, wr_greedy, "blended", blended_kpi_greedy_weight)
        score_primary = _kpi_score(wr_random, wr_greedy, args.primary_kpi, blended_kpi_greedy_weight)
        behavior = behavior_probe(
            ckpt,
            games=args.behavior_games,
            seed=args.seed + 20_000 + trained_steps // 1000,
        )

        improved_random = wr_random > best_random
        if improved_random:
            best_random = wr_random
            best_random_checkpoint = ckpt
            model.save(model_dir / "best_by_random")
        improved_greedy = wr_greedy > best_greedy
        if improved_greedy:
            best_greedy = wr_greedy
            best_greedy_checkpoint = ckpt
            model.save(model_dir / "best_by_greedy")
        improved_blended = score_blended > best_blended
        if improved_blended:
            best_blended = score_blended
            best_blended_checkpoint = ckpt
            model.save(model_dir / "best_by_blended")
        improved_primary = score_primary > best_primary
        if improved_primary:
            best_primary = score_primary
            best_primary_checkpoint = ckpt
            model.save(model_dir / "best_by_primary")

        rec = {
            "steps": trained_steps,
            "win_rate_random_stochastic": wr_random,
            "win_rate_greedy_stochastic": wr_greedy,
            "score_blended": score_blended,
            "score_primary": score_primary,
            "primary_kpi": args.primary_kpi,
            "league_opponent_prob": model.league_opponent_prob,
            "is_new_best_random": improved_random,
            "is_new_best_greedy": improved_greedy,
            "is_new_best_blended": improved_blended,
            "is_new_best_primary": improved_primary,
            "best_random_so_far": best_random,
            "best_random_checkpoint_so_far": best_random_checkpoint,
            "best_greedy_so_far": best_greedy,
            "best_greedy_checkpoint_so_far": best_greedy_checkpoint,
            "best_blended_so_far": best_blended,
            "best_blended_checkpoint_so_far": best_blended_checkpoint,
            "best_primary_so_far": best_primary,
            "best_primary_checkpoint_so_far": best_primary_checkpoint,
            "behavior": behavior,
        }
        records.append(rec)
        print(json.dumps(rec))

    with open(model_dir / "gated_metrics.json", "w") as f:
        json.dump(records, f, indent=2)

    summary = {
        "best_checkpoint": best_primary_checkpoint,
        "best_primary_checkpoint": best_primary_checkpoint,
        "best_random_checkpoint": best_random_checkpoint,
        "best_greedy_checkpoint": best_greedy_checkpoint,
        "best_blended_checkpoint": best_blended_checkpoint,
        "primary_kpi": args.primary_kpi,
        "best_primary_score": best_primary,
        "best_random_win_rate": best_random,
        "best_greedy_win_rate": best_greedy,
        "best_blended_score": best_blended,
        "max_steps": args.max_steps,
        "chunk_steps": args.chunk_steps,
        "eval_games": args.eval_games,
        "training_config": {
            "reward": args.reward,
            "n_envs": args.n_envs,
            "games_per_episode": args.games_per_episode,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "clip_range": args.clip_range,
            "ent_coef": args.ent_coef,
            "league_opponent_prob_start": league_prob_start,
            "league_opponent_prob_end": league_prob_end,
            "league_opponent_prob_ramp_steps": league_prob_ramp_steps,
            "anneal_lr": not args.no_anneal_lr,
            "anneal_clip": not args.no_anneal_clip,
            "blended_kpi_greedy_weight": blended_kpi_greedy_weight,
        },
    }
    with open(model_dir / "gated_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
