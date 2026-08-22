"""Targets and losses for a training-only future-safety-event teacher."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def future_event_targets(
    events: torch.Tensor,
    horizons: tuple[int, ...],
    *,
    terminal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Label whether an event occurs after each action within each horizon.

    The current event is excluded because the online guard has already
    preempted that action.  Censored suffixes are invalid; a true terminal makes
    their absence of future events observable.
    """
    if events.ndim != 1 or events.dtype != torch.bool:
        raise ValueError("events must be a one-dimensional boolean tensor")
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must be positive")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be unique")
    length = len(events)
    targets = torch.zeros(length, len(horizons), dtype=torch.float32)
    valid = torch.zeros(length, len(horizons), dtype=torch.bool)
    for index in range(length):
        for column, horizon in enumerate(horizons):
            end = min(index + horizon + 1, length)
            targets[index, column] = float(torch.any(events[index + 1 : end]))
            valid[index, column] = terminal or index + horizon < length
    return targets, valid


def balanced_event_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    *,
    maximum_positive_weight: float = 50.0,
) -> torch.Tensor:
    """Balanced BCE for sparse multi-horizon event labels."""
    if logits.shape != targets.shape or logits.shape != valid.shape:
        raise ValueError("logits, targets, and valid must have identical shapes")
    if not torch.any(valid):
        raise ValueError("event batch has no valid labels")
    if not math.isfinite(maximum_positive_weight) or maximum_positive_weight < 1.0:
        raise ValueError("maximum_positive_weight must be finite and at least one")
    weights = []
    for column in range(logits.shape[-1]):
        column_valid = valid[:, column]
        positives = targets[column_valid, column].sum()
        negatives = column_valid.sum() - positives
        if float(positives) == 0.0:
            weights.append(torch.ones((), dtype=logits.dtype, device=logits.device))
        else:
            weights.append((negatives / positives).clamp(1.0, maximum_positive_weight))
    positive_weight = torch.stack(weights)
    losses = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=positive_weight,
        reduction="none",
    )
    return losses[valid].mean()


def risk_adjusted_distillation_loss(
    student_logits: torch.Tensor,
    behavior_logits: torch.Tensor,
    action_masks: torch.Tensor,
    action_risk: torch.Tensor,
    *,
    risk_scale: float,
    anchor_coefficient: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Distill a risk-penalized teacher while anchoring the behavior policy."""
    if student_logits.shape != behavior_logits.shape:
        raise ValueError("student and behavior logits must match")
    if student_logits.shape != action_masks.shape or action_risk.shape != student_logits.shape:
        raise ValueError("masks and action risk must match logits")
    if not torch.all(action_masks.any(dim=-1)):
        raise ValueError("every state must retain at least one action")
    if not math.isfinite(risk_scale) or risk_scale < 0.0:
        raise ValueError("risk_scale must be finite and non-negative")
    if not math.isfinite(anchor_coefficient) or anchor_coefficient < 0.0:
        raise ValueError("anchor_coefficient must be finite and non-negative")
    if not torch.isfinite(action_risk).all():
        raise ValueError("action risk must be finite")

    invalid_logit = torch.finfo(student_logits.dtype).min
    student_log_probabilities = F.log_softmax(
        student_logits.masked_fill(~action_masks, invalid_logit), dim=-1
    )
    with torch.no_grad():
        behavior_log_probabilities = F.log_softmax(
            behavior_logits.masked_fill(~action_masks, invalid_logit), dim=-1
        )
        behavior_probabilities = behavior_log_probabilities.exp()
        teacher_log_probabilities = F.log_softmax(
            (behavior_logits - risk_scale * action_risk).masked_fill(
                ~action_masks, invalid_logit
            ),
            dim=-1,
        )
        teacher_probabilities = teacher_log_probabilities.exp()

    teacher_kl = torch.where(
        action_masks,
        teacher_probabilities
        * (teacher_log_probabilities - student_log_probabilities),
        torch.zeros_like(teacher_probabilities),
    ).sum(dim=-1).mean()
    anchor_kl = torch.where(
        action_masks,
        behavior_probabilities
        * (behavior_log_probabilities - student_log_probabilities),
        torch.zeros_like(behavior_probabilities),
    ).sum(dim=-1).mean()
    entropy = -torch.where(
        action_masks,
        student_log_probabilities.exp() * student_log_probabilities,
        torch.zeros_like(student_log_probabilities),
    ).sum(dim=-1).mean()
    loss = teacher_kl + anchor_coefficient * anchor_kl
    return loss, {
        "teacher_kl": teacher_kl.detach(),
        "anchor_kl": anchor_kl.detach(),
        "entropy": entropy.detach(),
        "mean_action_risk": action_risk.mean().detach(),
    }


__all__ = [
    "balanced_event_loss",
    "future_event_targets",
    "risk_adjusted_distillation_loss",
]
