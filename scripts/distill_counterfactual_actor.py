#!/usr/bin/env python3
"""Distill exact grouped collision branches into only the online actor head."""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from pc98rl.checkpoints import sha256_file
from pc98rl.counterfactual import (
    CounterfactualGroup,
    counterfactual_policy_metrics,
    load_counterfactual_group,
    validate_disjoint_groups,
)
from pc98rl.env import TH05_KINEMATICS
from pc98rl.model import EntityActorCritic
from pc98rl.offline_risk import risk_adjusted_distillation_loss


def _paths(patterns: list[str]) -> list[Path]:
    paths = [Path(path) for pattern in patterns for path in glob.glob(pattern)]
    return sorted(set(paths))


def _evaluate(
    actor: torch.nn.Module,
    group: CounterfactualGroup,
    *,
    risk_scale: float,
    anchor_coefficient: float,
) -> dict[str, float]:
    actor.eval()
    with torch.no_grad():
        logits = actor(group.actor_features)
        loss, components = risk_adjusted_distillation_loss(
            logits,
            group.behavior_logits,
            group.action_masks,
            group.collision_risk,
            risk_scale=risk_scale,
            anchor_coefficient=anchor_coefficient,
        )
        metrics = counterfactual_policy_metrics(logits, group)
    return {
        "selection_objective": float(loss),
        **metrics,
        "teacher_kl": float(components["teacher_kl"]),
        "anchor_kl": float(components["anchor_kl"]),
        "entropy": float(components["entropy"]),
    }


def _rounded(metrics: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 8) for key, value in metrics.items()}


def _checkpoint(
    *,
    saved: dict,
    model: EntityActorCritic,
    optimizer: torch.optim.Optimizer,
    source: Path,
    epoch: int,
    train: CounterfactualGroup,
    selection: CounterfactualGroup,
    args: argparse.Namespace,
) -> dict:
    checkpoint_args = dict(saved.get("args", {}))
    checkpoint_args.update(
        {
            "allow_bombs": False,
            "deathbomb_safety": False,
            "emergency_bomb_clearance": 0.0,
            "regular_bullet_safety_horizon": 6,
            "regular_bullet_safety_margin": 0.0,
            "regular_bullet_least_risk_fallback": False,
        }
    )
    return {
        **saved,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": checkpoint_args,
        "counterfactual_distillation": {
            "source": str(source.resolve()),
            "source_sha256": sha256_file(source),
            "epoch": epoch,
            "train_trajectories": [str(path) for path in train.paths],
            "selection_trajectories": [str(path) for path in selection.paths],
            "train_trajectory_ids": list(train.trajectory_ids),
            "selection_trajectory_ids": list(selection.trajectory_ids),
            "risk_scale": args.risk_scale,
            "anchor_coefficient": args.anchor_coefficient,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "trainable_module": "actor",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-trajectories", nargs="+", required=True)
    parser.add_argument("--selection-trajectories", nargs="+", required=True)
    parser.add_argument("--heldout-trajectories", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--student-output",
        required=True,
        help="best non-zero actor update retained only for diagnostic A/B",
    )
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--risk-scale", type=float, default=3.0)
    parser.add_argument("--anchor-coefficient", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    if args.learning_rate <= 0.0:
        parser.error("learning rate must be positive")

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    groups = {}
    for name, patterns in (
        ("train", args.train_trajectories),
        ("selection", args.selection_trajectories),
        ("heldout", args.heldout_trajectories),
    ):
        paths = _paths(patterns)
        if not paths:
            parser.error(f"no {name} trajectories matched")
        try:
            groups[name] = load_counterfactual_group(paths)
        except ValueError as error:
            parser.error(str(error))
    try:
        validate_disjoint_groups(**groups)
    except ValueError as error:
        parser.error(str(error))

    source = Path(args.checkpoint).expanduser().resolve()
    source_sha256 = sha256_file(source)
    if groups["train"].checkpoint_sha256 != source_sha256:
        parser.error("trajectory source hash does not match --checkpoint")
    saved = torch.load(source, map_location="cpu", weights_only=False)
    analytic_geometry = bool(saved.get("args", {}).get("analytic_geometry", False))
    model = EntityActorCritic(
        analytic_geometry=analytic_geometry,
        kinematic_spec=TH05_KINEMATICS if analytic_geometry else None,
    )
    model.load_state_dict(saved["model"])
    source_state = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.actor.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=args.learning_rate)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    student_output = Path(args.student_output)
    student_output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_dir = Path(args.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    if any(snapshot_dir.iterdir()):
        parser.error("snapshot directory must be empty")
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    baseline = {
        name: _rounded(
            _evaluate(
                model.actor,
                group,
                risk_scale=args.risk_scale,
                anchor_coefficient=args.anchor_coefficient,
            )
        )
        for name, group in groups.items()
    }
    records = []
    train = groups["train"]
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        baseline_record = {
            "phase": "counterfactual_actor",
            "epoch": 0,
            "train": baseline["train"],
            "selection": baseline["selection"],
            "wall_s": round(time.perf_counter() - started, 4),
        }
        metrics_file.write(json.dumps(baseline_record, sort_keys=True) + "\n")
        print(json.dumps(baseline_record, sort_keys=True), flush=True)

        for epoch in range(1, args.epochs + 1):
            model.actor.train()
            permutation = torch.randperm(train.anchors, generator=generator)
            for start in range(0, train.anchors, args.batch_size):
                indices = permutation[start : start + args.batch_size]
                logits = model.actor(train.actor_features[indices])
                loss, _ = risk_adjusted_distillation_loss(
                    logits,
                    train.behavior_logits[indices],
                    train.action_masks[indices],
                    train.collision_risk[indices],
                    risk_scale=args.risk_scale,
                    anchor_coefficient=args.anchor_coefficient,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
                optimizer.step()

            train_metrics = _rounded(
                _evaluate(
                    model.actor,
                    train,
                    risk_scale=args.risk_scale,
                    anchor_coefficient=args.anchor_coefficient,
                )
            )
            selection_metrics = _rounded(
                _evaluate(
                    model.actor,
                    groups["selection"],
                    risk_scale=args.risk_scale,
                    anchor_coefficient=args.anchor_coefficient,
                )
            )
            snapshot = snapshot_dir / f"epoch_{epoch:03d}.pt"
            torch.save(
                _checkpoint(
                    saved=saved,
                    model=model,
                    optimizer=optimizer,
                    source=source,
                    epoch=epoch,
                    train=train,
                    selection=groups["selection"],
                    args=args,
                ),
                snapshot,
            )
            record = {
                "phase": "counterfactual_actor",
                "epoch": epoch,
                "train": train_metrics,
                "selection": selection_metrics,
                "snapshot": str(snapshot.resolve()),
                "wall_s": round(time.perf_counter() - started, 4),
            }
            records.append(record)
            metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
            metrics_file.flush()
            print(json.dumps(record, sort_keys=True), flush=True)

    student = min(
        records,
        key=lambda record: (
            record["selection"]["selection_objective"],
            record["selection"]["expected_collision_risk"],
            record["epoch"],
        ),
    )
    baseline_candidate = {
        "epoch": 0,
        "snapshot": str(source),
        "train": baseline["train"],
        "selection": baseline["selection"],
    }
    selected = min(
        [baseline_candidate, *records],
        key=lambda record: (
            record["selection"]["selection_objective"],
            record["selection"]["expected_collision_risk"],
            record["epoch"],
        ),
    )
    shutil.copy2(selected["snapshot"], output)
    shutil.copy2(student["snapshot"], student_output)

    student_saved = torch.load(student_output, map_location="cpu", weights_only=False)
    model.load_state_dict(student_saved["model"])
    student_heldout = _rounded(
        _evaluate(
            model.actor,
            groups["heldout"],
            risk_scale=args.risk_scale,
            anchor_coefficient=args.anchor_coefficient,
        )
    )
    changed = []
    for name, tensor in model.state_dict().items():
        if not torch.equal(tensor, source_state[name]):
            changed.append(name)
            if not name.startswith("actor."):
                raise RuntimeError(f"non-actor parameter changed: {name}")
    if not changed:
        raise RuntimeError("distillation did not change any actor parameter")

    selected_saved = torch.load(output, map_location="cpu", weights_only=False)
    model.load_state_dict(selected_saved["model"])
    selected_heldout = _rounded(
        _evaluate(
            model.actor,
            groups["heldout"],
            risk_scale=args.risk_scale,
            anchor_coefficient=args.anchor_coefficient,
        )
    )
    report = {
        "research_question": (
            "Can exact short-horizon action-contrastive labels improve the frozen "
            "online actor without changing its encoder, recurrent state, critic, or latency?"
        ),
        "cpu_only": True,
        "nmnb": True,
        "source_checkpoint": str(source),
        "source_sha256": source_sha256,
        "split_unit": "complete trajectory",
        "groups": {
            name: {
                "trajectory_ids": list(group.trajectory_ids),
                "trajectory_count": len(group.trajectory_ids),
                "anchors": group.anchors,
            }
            for name, group in groups.items()
        },
        "configuration": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "risk_scale": args.risk_scale,
            "anchor_coefficient": args.anchor_coefficient,
            "seed": args.seed,
        },
        "trainable_module": "actor head only",
        "frozen_modules": ["entity encoder", "GRU", "critic"],
        "changed_parameter_names": changed,
        "baseline": baseline,
        "selected_epoch": selected["epoch"],
        "selected_snapshot": selected["snapshot"],
        "selected_output": str(output.resolve()),
        "selected_train": selected["train"],
        "selected_selection": selected["selection"],
        "selected_heldout": selected_heldout,
        "selection_decision": (
            "reject actor update" if selected["epoch"] == 0 else "retain actor update"
        ),
        "experimental_student_epoch": student["epoch"],
        "experimental_student_output": str(student_output.resolve()),
        "experimental_student_train": student["train"],
        "experimental_student_selection": student["selection"],
        "experimental_student_heldout": student_heldout,
        "heldout_opened_after_selection": True,
        "wall_s": round(time.perf_counter() - started, 4),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
