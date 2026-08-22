#!/usr/bin/env python3
"""Improve the small online actor from complete positive and negative attempts."""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pc98rl.distill import (
    conservative_outcome_loss,
    standardized_outcome_advantages,
    trajectory_outcome_score,
)
from pc98rl.env import TH05_KINEMATICS
from pc98rl.model import EntityActorCritic


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(trajectory: np.lib.npyio.NpzFile, key: str) -> int | bool:
    if key not in trajectory:
        raise ValueError(f"trajectory is missing required outcome field {key!r}")
    return trajectory[key].item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trajectories", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--clip-ratio", type=float, default=0.15)
    parser.add_argument("--anchor-coefficient", type=float, default=0.5)
    parser.add_argument("--completion-reward", type=float, default=1.0)
    parser.add_argument("--death-penalty", type=float, default=1.0)
    parser.add_argument("--bomb-penalty", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    torch.set_num_threads(8)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    checkpoint_path = Path(args.checkpoint)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    analytic_geometry = bool(saved.get("args", {}).get("analytic_geometry", False))
    model = EntityActorCritic(
        analytic_geometry=analytic_geometry,
        kinematic_spec=TH05_KINEMATICS if analytic_geometry else None,
    )
    model.load_state_dict(saved["model"])
    behavior_actor = copy.deepcopy(model.actor).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.actor.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=args.learning_rate)

    paths = sorted(
        Path(path) for pattern in args.trajectories for path in glob.glob(pattern)
    )
    if len(paths) < 2:
        parser.error("at least two trajectories must match")
    scores = []
    outcomes = []
    for path in paths:
        with np.load(path) as trajectory:
            outcome = {
                "no_miss_success": bool(_scalar(trajectory, "no_miss_success")),
                "deaths": int(_scalar(trajectory, "deaths")),
                "bombs_used": int(_scalar(trajectory, "bombs_used")),
            }
        score = trajectory_outcome_score(
            **outcome,
            completion_reward=args.completion_reward,
            death_penalty=args.death_penalty,
            bomb_penalty=args.bomb_penalty,
        )
        scores.append(score)
        outcomes.append(outcome)
    try:
        advantages = standardized_outcome_advantages(torch.tensor(scores)).numpy()
    except ValueError as error:
        parser.error(str(error))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_dir = Path(args.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_summary = {
        "trajectories": len(paths),
        "no_miss_successes": sum(outcome["no_miss_success"] for outcome in outcomes),
        "deaths": sum(outcome["deaths"] for outcome in outcomes),
        "mean_score": round(float(np.mean(scores)), 6),
        "score_standard_deviation": round(float(np.std(scores)), 6),
    }
    print(json.dumps({"dataset": dataset_summary}, sort_keys=True), flush=True)

    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        metric_names = (
            "loss",
            "policy_loss",
            "anchor_kl",
            "entropy",
            "mean_ratio",
            "clip_fraction",
        )
        totals = {key: 0.0 for key in metric_names}
        transitions = 0
        order = rng.permutation(len(paths))
        model.actor.train()
        for index in order:
            trajectory_index = int(index)
            path = paths[trajectory_index]
            with np.load(path) as trajectory:
                observations = torch.from_numpy(trajectory["observations"])
                actions = torch.from_numpy(trajectory["actions"])
                action_masks = torch.from_numpy(trajectory["action_masks"])
                valid = torch.from_numpy(trajectory["policy_valid"])
            with torch.no_grad():
                recurrent = model.recurrent_sequence(
                    observations.unsqueeze(0),
                    torch.zeros(1, 1, model.hidden_size),
                    torch.zeros(1, len(observations)),
                )
                behavior_logits = behavior_actor(recurrent)
            student_logits = model.actor(recurrent)
            episode_advantage = torch.full(
                actions.shape, float(advantages[trajectory_index])
            )
            loss, components = conservative_outcome_loss(
                student_logits,
                behavior_logits,
                actions,
                action_masks,
                valid,
                episode_advantage,
                clip_ratio=args.clip_ratio,
                anchor_coefficient=args.anchor_coefficient,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
            optimizer.step()
            count = int(valid.sum().item())
            transitions += count
            totals["loss"] += float(loss.item()) * count
            for key, value in components.items():
                totals[key] += float(value.item()) * count

        metrics = {
            "epoch": epoch,
            "transitions": transitions,
            "learning_rate": args.learning_rate,
            "clip_ratio": args.clip_ratio,
            "anchor_coefficient": args.anchor_coefficient,
            **dataset_summary,
            **{key: round(value / transitions, 6) for key, value in totals.items()},
            "wall_s": round(time.perf_counter() - started, 3),
        }
        print(json.dumps(metrics, sort_keys=True), flush=True)
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(metrics, sort_keys=True) + "\n")

        checkpoint_args = dict(saved.get("args", {}))
        checkpoint_args.update(
            {
                "deathbomb_safety": True,
                "regular_bullet_safety_horizon": 6,
                "regular_bullet_safety_margin": 0.0,
            }
        )
        improved = {
            **saved,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": checkpoint_args,
            "outcome_improvement": {
                "source": str(checkpoint_path.resolve()),
                "source_sha256": _sha256(checkpoint_path),
                "epoch": epoch,
                "trajectories": [str(path.resolve()) for path in paths],
                "scores": scores,
                "advantages": advantages.tolist(),
                "clip_ratio": args.clip_ratio,
                "anchor_coefficient": args.anchor_coefficient,
                "completion_reward": args.completion_reward,
                "death_penalty": args.death_penalty,
                "bomb_penalty": args.bomb_penalty,
            },
        }
        torch.save(improved, output)
        torch.save(improved, snapshot_dir / f"epoch_{epoch:03d}.pt")


if __name__ == "__main__":
    main()
