#!/usr/bin/env python3
"""Train an offline action-risk teacher and distill it into the online actor."""

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

from pc98rl.env import TH05_KINEMATICS
from pc98rl.model import EntityActorCritic, FutureMissHead
from pc98rl.offline_risk import (
    balanced_event_loss,
    future_event_targets,
    risk_adjusted_distillation_loss,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(patterns: list[str]) -> list[Path]:
    return sorted(Path(path) for pattern in patterns for path in glob.glob(pattern))


def _recurrent(model: EntityActorCritic, observations: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.recurrent_sequence(
            observations.unsqueeze(0),
            torch.zeros(1, 1, model.hidden_size),
            torch.zeros(1, len(observations)),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trajectories", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[8, 24, 64])
    parser.add_argument("--risk-epochs", type=int, default=5)
    parser.add_argument("--actor-epochs", type=int, default=5)
    parser.add_argument("--risk-learning-rate", type=float, default=3e-4)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-4)
    parser.add_argument("--risk-scale", type=float, default=3.0)
    parser.add_argument("--anchor-coefficient", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    if args.risk_epochs < 1 or args.actor_epochs < 1:
        parser.error("risk and actor epochs must be positive")
    horizons = tuple(args.horizons)
    if not horizons or any(horizon < 1 for horizon in horizons):
        parser.error("horizons must be positive")

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

    paths = _paths(args.trajectories)
    if not paths:
        parser.error("no trajectories matched")
    required_fields = {
        "observations",
        "actions",
        "action_masks",
        "policy_valid",
        "safety_events",
        "miss_events",
        "terminal",
    }
    for path in paths:
        with np.load(path) as trajectory:
            missing = required_fields.difference(trajectory.files)
        if missing:
            parser.error(f"{path} is missing fields: {sorted(missing)}")

    risk_head = FutureMissHead(model.hidden_size, 19, len(horizons))
    risk_optimizer = torch.optim.Adam(
        risk_head.parameters(), lr=args.risk_learning_rate
    )
    actor_optimizer = torch.optim.Adam(
        model.actor.parameters(), lr=args.actor_learning_rate
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_dir = Path(args.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    for epoch in range(1, args.risk_epochs + 1):
        totals = {
            "loss": 0.0,
            "positive_probability": 0.0,
            "negative_probability": 0.0,
        }
        valid_labels = 0
        positive_labels = 0
        negative_labels = 0
        risk_head.train()
        for index in rng.permutation(len(paths)):
            path = paths[int(index)]
            with np.load(path) as trajectory:
                observations = torch.from_numpy(trajectory["observations"])
                actions = torch.from_numpy(trajectory["actions"])
                policy_valid = torch.from_numpy(trajectory["policy_valid"])
                events = torch.from_numpy(
                    trajectory["safety_events"] | trajectory["miss_events"]
                )
                terminal = bool(trajectory["terminal"])
            recurrent = _recurrent(model, observations)
            targets, label_valid = future_event_targets(
                events, horizons, terminal=terminal
            )
            label_valid &= policy_valid.unsqueeze(-1)
            logits = risk_head(recurrent, actions)
            loss = balanced_event_loss(logits, targets, label_valid)
            risk_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(risk_head.parameters(), 1.0)
            risk_optimizer.step()

            with torch.no_grad():
                probabilities = logits.sigmoid()
                positive = label_valid & (targets > 0.5)
                negative = label_valid & ~positive
                positive_count = int(positive.sum())
                negative_count = int(negative.sum())
                count = positive_count + negative_count
                valid_labels += count
                positive_labels += positive_count
                negative_labels += negative_count
                totals["loss"] += float(loss.detach()) * count
                totals["positive_probability"] += float(
                    probabilities[positive].sum()
                )
                totals["negative_probability"] += float(
                    probabilities[negative].sum()
                )

        metrics = {
            "phase": "risk_teacher",
            "epoch": epoch,
            "trajectories": len(paths),
            "horizons": list(horizons),
            "valid_labels": valid_labels,
            "positive_labels": positive_labels,
            "positive_fraction": round(positive_labels / valid_labels, 6),
            "loss": round(totals["loss"] / valid_labels, 6),
            "positive_probability": (
                round(totals["positive_probability"] / positive_labels, 6)
                if positive_labels
                else None
            ),
            "negative_probability": (
                round(totals["negative_probability"] / negative_labels, 6)
                if negative_labels
                else None
            ),
            "wall_s": round(time.perf_counter() - started, 3),
        }
        print(json.dumps(metrics, sort_keys=True), flush=True)
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(metrics, sort_keys=True) + "\n")

    risk_head.eval()
    action_ids = torch.arange(19)
    for epoch in range(1, args.actor_epochs + 1):
        totals = {
            "loss": 0.0,
            "teacher_kl": 0.0,
            "anchor_kl": 0.0,
            "entropy": 0.0,
            "mean_action_risk": 0.0,
        }
        transitions = 0
        model.actor.train()
        for index in rng.permutation(len(paths)):
            path = paths[int(index)]
            with np.load(path) as trajectory:
                observations = torch.from_numpy(trajectory["observations"])
                action_masks = torch.from_numpy(trajectory["action_masks"])
            recurrent = _recurrent(model, observations)
            with torch.no_grad():
                behavior_logits = behavior_actor(recurrent)
                candidate_recurrent = recurrent.unsqueeze(1).expand(-1, 19, -1)
                candidate_actions = action_ids.unsqueeze(0).expand(len(recurrent), -1)
                risk_logits = risk_head(
                    candidate_recurrent.reshape(-1, model.hidden_size),
                    candidate_actions.reshape(-1),
                ).reshape(len(recurrent), 19, len(horizons))
                action_risk = risk_logits[..., -1].sigmoid()
            student_logits = model.actor(recurrent)
            loss, components = risk_adjusted_distillation_loss(
                student_logits,
                behavior_logits,
                action_masks,
                action_risk,
                risk_scale=args.risk_scale,
                anchor_coefficient=args.anchor_coefficient,
            )
            actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
            actor_optimizer.step()
            count = len(observations)
            transitions += count
            totals["loss"] += float(loss.detach()) * count
            for key, value in components.items():
                totals[key] += float(value) * count

        metrics = {
            "phase": "actor_distillation",
            "epoch": epoch,
            "trajectories": len(paths),
            "transitions": transitions,
            "risk_scale": args.risk_scale,
            "anchor_coefficient": args.anchor_coefficient,
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
            "optimizer": actor_optimizer.state_dict(),
            "args": checkpoint_args,
            "future_safety_distillation": {
                "source": str(checkpoint_path.resolve()),
                "source_sha256": _sha256(checkpoint_path),
                "epoch": epoch,
                "trajectories": [str(path.resolve()) for path in paths],
                "horizons": list(horizons),
                "risk_scale": args.risk_scale,
                "anchor_coefficient": args.anchor_coefficient,
                "risk_head": risk_head.state_dict(),
            },
        }
        torch.save(distilled, output)
        torch.save(distilled, snapshot_dir / f"epoch_{epoch:03d}.pt")


if __name__ == "__main__":
    main()
