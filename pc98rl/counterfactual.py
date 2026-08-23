"""Grouped datasets and metrics for offline counterfactual actor distillation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


DATASET_FORMAT = "pc98rl-counterfactual-trajectory-v2-nmnb"
REQUIRED_FIELDS = {
    "format",
    "trajectory_id",
    "checkpoint_sha256",
    "actor_features",
    "behavior_logits",
    "action_masks",
    "collision_risk",
}


@dataclass(frozen=True)
class CounterfactualGroup:
    paths: tuple[Path, ...]
    trajectory_ids: tuple[str, ...]
    checkpoint_sha256: str
    actor_features: torch.Tensor
    behavior_logits: torch.Tensor
    action_masks: torch.Tensor
    collision_risk: torch.Tensor

    @property
    def anchors(self) -> int:
        return int(self.actor_features.shape[0])


def load_counterfactual_group(paths: list[Path]) -> CounterfactualGroup:
    """Load complete trajectories and reject malformed or duplicate groups."""
    resolved = tuple(path.expanduser().resolve() for path in paths)
    if not resolved:
        raise ValueError("counterfactual group must contain at least one trajectory")
    if len(set(resolved)) != len(resolved):
        raise ValueError("counterfactual trajectory paths must be unique")

    trajectory_ids = []
    checkpoint_hashes = []
    features = []
    behavior_logits = []
    action_masks = []
    collision_risk = []
    for path in resolved:
        with np.load(path, allow_pickle=False) as trajectory:
            missing = REQUIRED_FIELDS.difference(trajectory.files)
            if missing:
                raise ValueError(f"{path} is missing fields: {sorted(missing)}")
            if str(trajectory["format"]) != DATASET_FORMAT:
                raise ValueError(f"{path} has an unsupported dataset format")
            trajectory_id = str(trajectory["trajectory_id"])
            checkpoint_hash = str(trajectory["checkpoint_sha256"])
            feature = np.asarray(trajectory["actor_features"], dtype=np.float32)
            logits = np.asarray(trajectory["behavior_logits"], dtype=np.float32)
            masks = np.asarray(trajectory["action_masks"], dtype=np.bool_)
            risk = np.asarray(trajectory["collision_risk"], dtype=np.float32)

        if feature.ndim != 2 or feature.shape[1] != 128:
            raise ValueError(f"{path} actor features must have shape [N, 128]")
        expected = (feature.shape[0], 19)
        if logits.shape != expected or masks.shape != expected or risk.shape != expected:
            raise ValueError(f"{path} action arrays must have shape {expected}")
        if feature.shape[0] < 1:
            raise ValueError(f"{path} has no anchors")
        if not np.all(masks.any(axis=1)):
            raise ValueError(f"{path} contains an anchor without a legal action")
        if np.any(masks[:, 18]):
            raise ValueError(f"{path} must exclude the unbranched bomb action")
        if not np.isfinite(feature).all() or not np.isfinite(logits).all():
            raise ValueError(f"{path} contains non-finite actor data")
        if not np.isfinite(risk).all() or np.any((risk < 0.0) | (risk > 1.0)):
            raise ValueError(f"{path} collision risk must be finite in [0, 1]")
        legal_risk = np.where(masks, risk, np.nan)
        if np.any(np.nansum(legal_risk, axis=1) == 0.0) or np.any(
            np.nansum(np.where(masks, 1.0 - risk, np.nan), axis=1) == 0.0
        ):
            raise ValueError(f"{path} contains a non-contrastive anchor")

        trajectory_ids.append(trajectory_id)
        checkpoint_hashes.append(checkpoint_hash)
        features.append(feature)
        behavior_logits.append(logits)
        action_masks.append(masks)
        collision_risk.append(risk)

    if len(set(trajectory_ids)) != len(trajectory_ids):
        raise ValueError("trajectory IDs must be unique within a group")
    if len(set(checkpoint_hashes)) != 1:
        raise ValueError("all trajectories must come from one source checkpoint")
    return CounterfactualGroup(
        paths=resolved,
        trajectory_ids=tuple(trajectory_ids),
        checkpoint_sha256=checkpoint_hashes[0],
        actor_features=torch.from_numpy(np.concatenate(features)),
        behavior_logits=torch.from_numpy(np.concatenate(behavior_logits)),
        action_masks=torch.from_numpy(np.concatenate(action_masks)),
        collision_risk=torch.from_numpy(np.concatenate(collision_risk)),
    )


def validate_disjoint_groups(**groups: CounterfactualGroup) -> None:
    """Enforce trajectory-level train/selection/held-out separation."""
    owners: dict[str, str] = {}
    for name, group in groups.items():
        for trajectory_id in group.trajectory_ids:
            if trajectory_id in owners:
                raise ValueError(
                    f"trajectory {trajectory_id!r} appears in both "
                    f"{owners[trajectory_id]} and {name}"
                )
            owners[trajectory_id] = name
    checkpoint_hashes = {group.checkpoint_sha256 for group in groups.values()}
    if len(checkpoint_hashes) != 1:
        raise ValueError("all groups must share one source checkpoint")


def counterfactual_policy_metrics(
    logits: torch.Tensor,
    group: CounterfactualGroup,
) -> dict[str, float]:
    """Measure causal collision mass and drift from the frozen behavior actor."""
    if logits.shape != group.behavior_logits.shape:
        raise ValueError("policy logits must match the counterfactual group")
    invalid = torch.finfo(logits.dtype).min
    log_probabilities = F.log_softmax(
        logits.masked_fill(~group.action_masks, invalid), dim=-1
    )
    probabilities = log_probabilities.exp()
    with torch.no_grad():
        behavior_log_probabilities = F.log_softmax(
            group.behavior_logits.masked_fill(~group.action_masks, invalid), dim=-1
        )
        behavior_probabilities = behavior_log_probabilities.exp()
    expected_risk = (probabilities * group.collision_risk).sum(dim=-1)
    actions = logits.masked_fill(~group.action_masks, invalid).argmax(dim=-1)
    argmax_risk = group.collision_risk.gather(1, actions.unsqueeze(1)).squeeze(1)
    behavior_kl = torch.where(
        group.action_masks,
        behavior_probabilities * (behavior_log_probabilities - log_probabilities),
        torch.zeros_like(probabilities),
    ).sum(dim=-1)
    return {
        "expected_collision_risk": float(expected_risk.mean()),
        "argmax_collision_fraction": float((argmax_risk > 0.5).float().mean()),
        "behavior_kl": float(behavior_kl.mean()),
    }


def balanced_binary_risk_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Give colliding and safe branch labels equal aggregate weight."""
    if logits.shape != targets.shape or logits.shape != valid.shape:
        raise ValueError("risk logits, targets, and valid mask must match")
    positive = valid & (targets > 0.5)
    negative = valid & ~positive
    if not torch.any(positive) or not torch.any(negative):
        raise ValueError("balanced risk loss requires both collision classes")
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return 0.5 * (losses[positive].mean() + losses[negative].mean())


def binary_risk_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, float]:
    """Return threshold and ranking metrics over legal counterfactual actions."""
    if probabilities.shape != targets.shape or probabilities.shape != valid.shape:
        raise ValueError("risk probabilities, targets, and valid mask must match")
    scores = probabilities[valid].detach().cpu().numpy().astype(np.float64)
    labels = (targets[valid].detach().cpu().numpy() > 0.5)
    positives = scores[labels]
    negatives = scores[~labels]
    if not len(positives) or not len(negatives):
        raise ValueError("risk metrics require both collision classes")
    auc = float(
        (
            (positives[:, None] > negatives[None, :]).sum()
            + 0.5 * (positives[:, None] == negatives[None, :]).sum()
        )
        / (len(positives) * len(negatives))
    )
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    average_precision = float(precision[sorted_labels].mean())
    predicted = scores >= 0.5
    true_positive_rate = float(predicted[labels].mean())
    true_negative_rate = float((~predicted[~labels]).mean())
    return {
        "roc_auc": auc,
        "average_precision": average_precision,
        "balanced_accuracy": 0.5 * (true_positive_rate + true_negative_rate),
        "brier": float(np.mean((scores - labels.astype(np.float64)) ** 2)),
        "mean_positive_probability": float(positives.mean()),
        "mean_negative_probability": float(negatives.mean()),
        "positive_fraction": float(labels.mean()),
        "labels": int(len(labels)),
    }


__all__ = [
    "CounterfactualGroup",
    "DATASET_FORMAT",
    "balanced_binary_risk_loss",
    "binary_risk_metrics",
    "counterfactual_policy_metrics",
    "load_counterfactual_group",
    "validate_disjoint_groups",
]
