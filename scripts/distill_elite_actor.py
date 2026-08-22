#!/usr/bin/env python3
"""Conservatively reinforce elite trajectories in the small online actor."""

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

from pc98rl.distill import conservative_distillation_loss, elite_episode_weight
from pc98rl.env import TH05_KINEMATICS
from pc98rl.model import EntityActorCritic


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trajectories", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--anchor-coefficient", type=float, default=0.5)
    parser.add_argument("--bomb-weight-decay", type=float, default=0.7)
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
    teacher_actor = copy.deepcopy(model.actor).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.actor.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=args.learning_rate)

    paths = sorted(
        Path(path)
        for pattern in args.trajectories
        for path in glob.glob(pattern)
    )
    if not paths:
        parser.error("no trajectories matched")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_dir = Path(args.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        totals = {"loss": 0.0, "elite_nll": 0.0, "anchor_kl": 0.0, "entropy": 0.0}
        transitions = 0
        order = rng.permutation(len(paths))
        model.actor.train()
        for index in order:
            path = paths[int(index)]
            with np.load(path) as trajectory:
                observations = torch.from_numpy(trajectory["observations"])
                actions = torch.from_numpy(trajectory["actions"])
                action_masks = torch.from_numpy(trajectory["action_masks"])
                valid = torch.from_numpy(trajectory["policy_valid"])
                bombs_used = int(trajectory["bombs_used"])
            with torch.no_grad():
                recurrent = model.recurrent_sequence(
                    observations.unsqueeze(0),
                    torch.zeros(1, 1, model.hidden_size),
                    torch.zeros(1, len(observations)),
                )
                teacher_logits = teacher_actor(recurrent)
            student_logits = model.actor(recurrent)
            loss, components = conservative_distillation_loss(
                student_logits,
                teacher_logits,
                actions,
                action_masks,
                valid,
                anchor_coefficient=args.anchor_coefficient,
            )
            weight = elite_episode_weight(bombs_used, args.bomb_weight_decay)
            optimizer.zero_grad(set_to_none=True)
            (weight * loss).backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
            optimizer.step()
            count = int(valid.sum().item())
            transitions += count
            totals["loss"] += float(loss.item()) * count
            for key, value in components.items():
                totals[key] += float(value.item()) * count

        metrics = {
            "epoch": epoch,
            "trajectories": len(paths),
            "transitions": transitions,
            "learning_rate": args.learning_rate,
            "anchor_coefficient": args.anchor_coefficient,
            "bomb_weight_decay": args.bomb_weight_decay,
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
        distilled = {
            **saved,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": checkpoint_args,
            "distillation": {
                "source": str(checkpoint_path.resolve()),
                "source_sha256": _sha256(checkpoint_path),
                "epoch": epoch,
                "trajectories": [str(path.resolve()) for path in paths],
                "anchor_coefficient": args.anchor_coefficient,
                "bomb_weight_decay": args.bomb_weight_decay,
            },
        }
        torch.save(distilled, output)
        torch.save(distilled, snapshot_dir / f"epoch_{epoch:03d}.pt")


if __name__ == "__main__":
    main()
