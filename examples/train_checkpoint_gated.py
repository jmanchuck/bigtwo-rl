"""Train with stochastic checkpoint gating to prevent effective KPI regression.

This script trains in chunks, evaluates each checkpoint against random/greedy,
and promotes only the best-by-random checkpoint.
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
    parser.add_argument("--league-opponent-prob", type=float, default=0.0)
    parser.add_argument(
        "--reward",
        choices=["default", "sparse", "progressive", "ranking", "score_margin"],
        default="default",
    )
    parser.add_argument("--no-anneal-lr", action="store_true")
    parser.add_argument("--no-anneal-clip", action="store_true")
    args = parser.parse_args()

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
        league_opponent_prob=args.league_opponent_prob,
        anneal_lr=not args.no_anneal_lr,
        anneal_clip=not args.no_anneal_clip,
    )

    model = build_model(trainer, log_dir)
    callback = CallbackList([SimpleSelfPlayCallback(verbose=0)])

    trained_steps = 0
    best_random = -1.0
    best_checkpoint = None
    records: list[dict] = []

    while trained_steps < args.max_steps:
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
        behavior = behavior_probe(
            ckpt,
            games=args.behavior_games,
            seed=args.seed + 20_000 + trained_steps // 1000,
        )

        improved = wr_random > best_random
        if improved:
            best_random = wr_random
            best_checkpoint = ckpt
            model.save(model_dir / "best_by_random")

        rec = {
            "steps": trained_steps,
            "win_rate_random_stochastic": wr_random,
            "win_rate_greedy_stochastic": wr_greedy,
            "is_new_best_random": improved,
            "best_random_so_far": best_random,
            "best_checkpoint_so_far": best_checkpoint,
            "behavior": behavior,
        }
        records.append(rec)
        print(json.dumps(rec))

    with open(model_dir / "gated_metrics.json", "w") as f:
        json.dump(records, f, indent=2)

    summary = {
        "best_checkpoint": best_checkpoint,
        "best_random_win_rate": best_random,
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
            "league_opponent_prob": args.league_opponent_prob,
            "anneal_lr": not args.no_anneal_lr,
            "anneal_clip": not args.no_anneal_clip,
        },
    }
    with open(model_dir / "gated_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
