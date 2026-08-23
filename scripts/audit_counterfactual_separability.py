#!/usr/bin/env python3
"""Audit whether frozen recurrent features support grouped action-risk transfer."""

from __future__ import annotations

import argparse
import copy
import glob
import json
import time
from pathlib import Path

import torch
from torch import nn

from pc98rl.counterfactual import (
    CounterfactualGroup,
    balanced_binary_risk_loss,
    binary_risk_metrics,
    load_counterfactual_group,
    validate_disjoint_groups,
)


def _paths(patterns: list[str]) -> list[Path]:
    return sorted(
        set(Path(path) for pattern in patterns for path in glob.glob(pattern))
    )


def _model(architecture: str) -> nn.Module:
    if architecture == "linear":
        return nn.Linear(128, 19)
    if architecture == "mlp":
        return nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 19))
    raise ValueError(f"unknown architecture: {architecture}")


def _evaluate(model: nn.Module, group: CounterfactualGroup) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(group.actor_features)
        loss = balanced_binary_risk_loss(
            logits, group.collision_risk, group.action_masks
        )
        metrics = binary_risk_metrics(
            logits.sigmoid(), group.collision_risk, group.action_masks
        )
    return {"balanced_bce": float(loss), **metrics}


def _round(metrics: dict[str, float]) -> dict[str, float | int]:
    return {
        key: value if isinstance(value, int) else round(value, 8)
        for key, value in metrics.items()
    }


def _constant_baseline(
    train: CounterfactualGroup, group: CounterfactualGroup
) -> dict[str, float]:
    positive = (train.collision_risk * train.action_masks).sum(dim=0)
    counts = train.action_masks.sum(dim=0)
    probabilities = (positive + 1.0) / (counts + 2.0)
    predictions = probabilities.unsqueeze(0).expand(group.anchors, -1)
    logits = torch.logit(predictions.clamp(1e-5, 1.0 - 1e-5))
    return {
        "balanced_bce": float(
            balanced_binary_risk_loss(
                logits, group.collision_risk, group.action_masks
            )
        ),
        **binary_risk_metrics(
            predictions, group.collision_risk, group.action_masks
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-trajectories", nargs="+", required=True)
    parser.add_argument("--selection-trajectories", nargs="+", required=True)
    parser.add_argument("--heldout-trajectories", nargs="+", required=True)
    parser.add_argument("--architectures", nargs="+", default=["linear", "mlp"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1901, 1902, 1903])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--report", required=True)
    parser.add_argument("--metrics", required=True)
    args = parser.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        parser.error("invalid optimization configuration")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if any(name not in {"linear", "mlp"} for name in args.architectures):
        parser.error("architectures must be linear or mlp")

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    groups = {
        name: load_counterfactual_group(_paths(patterns))
        for name, patterns in (
            ("train", args.train_trajectories),
            ("selection", args.selection_trajectories),
            ("heldout", args.heldout_trajectories),
        )
    }
    validate_disjoint_groups(**groups)
    started = time.perf_counter()
    candidates = []
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for architecture in args.architectures:
            for seed in args.seeds:
                torch.manual_seed(seed)
                model = _model(architecture)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay,
                )
                for epoch in range(1, args.epochs + 1):
                    model.train()
                    logits = model(groups["train"].actor_features)
                    loss = balanced_binary_risk_loss(
                        logits,
                        groups["train"].collision_risk,
                        groups["train"].action_masks,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    train_metrics = _round(_evaluate(model, groups["train"]))
                    selection_metrics = _round(
                        _evaluate(model, groups["selection"])
                    )
                    record = {
                        "architecture": architecture,
                        "seed": seed,
                        "epoch": epoch,
                        "train": train_metrics,
                        "selection": selection_metrics,
                        "wall_s": round(time.perf_counter() - started, 4),
                    }
                    metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                    candidates.append(
                        {
                            **record,
                            "state_dict": copy.deepcopy(model.state_dict()),
                        }
                    )

    selected = min(
        candidates,
        key=lambda candidate: (
            candidate["selection"]["balanced_bce"],
            -candidate["selection"]["roc_auc"],
            -candidate["selection"]["average_precision"],
            candidate["epoch"],
        ),
    )
    model = _model(selected["architecture"])
    model.load_state_dict(selected["state_dict"])
    heldout = _round(_evaluate(model, groups["heldout"]))
    constant = {
        name: _round(_constant_baseline(groups["train"], group))
        for name, group in groups.items()
    }
    supports_frozen_features = bool(
        heldout["roc_auc"] >= 0.65
        and heldout["balanced_bce"] < constant["heldout"]["balanced_bce"]
    )
    report = {
        "research_question": (
            "Are exact action-collision outcomes separable from the frozen source "
            "GRU state across trajectory-held-out NMNB anchors?"
        ),
        "cpu_only": True,
        "online_modules_added": 0,
        "split_unit": "complete trajectory",
        "groups": {
            name: {
                "trajectory_ids": list(group.trajectory_ids),
                "trajectories": len(group.trajectory_ids),
                "anchors": group.anchors,
            }
            for name, group in groups.items()
        },
        "configuration": {
            "architectures": args.architectures,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "constant_action_rate_baseline": constant,
        "selected": {
            key: selected[key]
            for key in ("architecture", "seed", "epoch", "train", "selection")
        },
        "selected_heldout": heldout,
        "heldout_opened_after_selection": True,
        "supports_frozen_feature_action_risk": supports_frozen_features,
        "decision": (
            "Scale exact data and use a richer offline teacher on frozen features."
            if supports_frozen_features
            else "Do not assume the frozen recurrent state supports transferable action risk."
        ),
        "wall_s": round(time.perf_counter() - started, 4),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
